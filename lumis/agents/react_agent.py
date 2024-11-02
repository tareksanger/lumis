from inspect import iscoroutinefunction
import logging
from typing import Any, Callable, Literal, Optional, Union

from lumis.agents.base.base_agent import BaseAgent
from lumis.llm.openai_llm import LLM
from lumis.memory import BaseMemory, SimpleMemory

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt


class ReActThought(BaseModel):
    """
    Represents a thought in the agent's reasoning process.
    """

    action: Literal["reason", "act", "finish"] = Field(..., description="The type of action to be performed.")
    thought: str = Field(..., description="Your thought regarding taking the action.")
    observations: str = Field(..., description="Your current observation.")


E = Literal[
    # The first action, called after the agents memory and messages have be configured, but before the first thought is generated.
    # Only called Once. This event is passed the agent as arguments.
    "initialize",
    # Called on every step before the next action is taken. This event is passed the agent and the thought as arguments.
    "step",
    # Called at the end of every step after the action is taken. This event is passed the agent as arguments..
    "after_step",
    # Called only when the agent want to act (Use a tool.). This is called before the action is taken. This event is passed the agent as arguments.
    "before_act",
    # Called only when the agents act is completed successfully. This event is passed the agent as arguments.
    "after_act",
    # Called only when the agents act has errored out. This event is passed the agent as arguments.
    "act_error",
    # Called only when the agent has completed its run. This event is passed the agent as arguments.
    "complete",
    # Called only when the agent has completely errored out terminating the run. This event is passed the agent as arguments.
    "error",
    # Called only when the agent has been reset after all memory has been removed and everything else. This event is passed the agent as arguments.
    "reset",
    # Max Steps is called when the agent reaches the maximum number of steps they can reach. This can be used to force a generation or an answer.
    "max_steps_reached",
]

DEFAULT_MAX_STEPS = 100


class ReactAgent(BaseAgent[E]):
    """
    A base class for agents that use a react-based approach to reasoning.
    """

    BASE_REACT_PROMPT: str = (
        "You will work step-by-step, alternating between ReActThought, Action, and Observation steps:\n\n"
        "Thought: You will reason through the current situation, ask yourself questions, describe your current understanding of the situation, and decide on the next action to take. "
        "Observation: The current observation.\n\n"
        "Action: You will decide to take one of the following actions:\n"
        "- reason: Continue reasoning about the current situation.\n"
        "- act: Perform an action using one of the tools in your toolkit (be specific about the action you wish to take).\n"
        "- finish: Move on to the next question, or finish the task if no more questions remain.\n\n"
        "Note: You have a limited number of steps you can take. You have taken {step_count} of {max_steps} steps. Once all steps have been run you will fail the task."
    )

    ACTIONS: str = "Here is the list of actions you can take:\n{}"

    def __init__(
        self,
        llm: Optional[LLM] = None,
        tools: list[Callable] = [],
        memory: BaseMemory = SimpleMemory(),
        parallel_tool_calls: bool = True,
        finish_condition_callback: Optional[Callable[[], Union[str, None]]] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
    ):
        super().__init__(llm=llm, tools=tools, memory=memory, verbose=verbose, logger=logger)
        self.parallel_tool_calls = parallel_tool_calls

        self.max_steps = DEFAULT_MAX_STEPS
        self.step_count = 0

        self.thought: ReActThought | None = None
        self.finish_condition_callback = finish_condition_callback
        self.has_initialized = False

    async def initialize(self, messages: list[ChatCompletionMessageParam] = []) -> Optional[ReActThought]:
        """
        Initializes the agent's reasoning process.

        Args:
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            Optional[ReActThought]: The initial ReActThought object or None if initialization fails.
        """
        self.logger.info("Initializing ReactAgent.")

        try:
            if not self.has_initialized:
                # Inject the default react agent messages with the config messages.
                # TODO: Add function to BaseMemory to insert memory items into the top of the list. This way users can modify memory however they wish before initializing the agent.
                default_messages = [
                    {"role": "system", "content": self.BASE_REACT_PROMPT.format(step_count=self.step_count, max_steps=self.max_steps)},
                    {"role": "system", "content": self.ACTIONS.format("\n".join([f"- {tool.__name__}" for tool in self.tools]))},
                    *messages,
                ]
                for message in default_messages:
                    self.memory.add(message)
                self.has_initialized = True

            await self.emit("initialize", self)
            initial_response = await self.llm.structured_completion(response_format=ReActThought, messages=self.memory.get())
            if initial_response:
                self.logger.info("Received initial thought from structured completion.")
                return initial_response.parsed
            else:
                self.logger.warning("No initial thought received during initialization.")
                return None

        except Exception as e:
            self.log_exception(e)
            self.logger.exception("Failed to initialize ReactAgent")
            return None

    async def step(self, thought: Optional[ReActThought]) -> Optional[ReActThought]:
        """
        Executes a single reasoning step based on the current ReActThought.

        Args:
            thought (Optional[ReActThought]): The current ReActThought object.

        Returns:
            Optional[ReActThought]: The next ReActThought object or None to terminate the loop.
        """
        self.memory.update(0, {"role": "system", "content": self.BASE_REACT_PROMPT.format(step_count=self.step_count, max_steps=self.max_steps)})

        if thought is None:
            self.logger.warning("The agent returned an invalid thought.")
            return None

        try:
            # Emit the step event before adding the message to memory.
            await self.emit("step", self)

            thought_message = f"\nAction: {thought.action}\nThought: {thought.thought}\n"
            if thought.observations:
                thought_message += f"\nObservation: {thought.observations}"

            self.add_message({"role": "assistant", "content": thought_message, "name": "lumis"})
            self.logger.debug(f"Added assistant message: {thought_message}")

            if thought.action == "act":
                self.logger.info("Performing action: act")
                await self.act()

            elif thought.action == "finish":
                block_finish = False
                if self.finish_condition_callback:
                    if iscoroutinefunction(self.finish_condition_callback):
                        block_finish_message = await self.finish_condition_callback()
                    else:
                        block_finish_message = self.finish_condition_callback()

                    if block_finish_message is not None:
                        self.add_message({"role": "system", "content": block_finish_message})
                        block_finish = True

                if not block_finish:
                    return None
                else:
                    self.logger.warning("The finish condition callback returned False. The agent will continue to run.")

            await self.emit("after_step", self)

            # Obtain the next thought from structured completion
            self.logger.debug("Requesting new thought from structured completion.")
            chat_message = await self.llm.structured_completion(response_format=ReActThought, messages=self.memory.get())
            if chat_message:
                self.logger.info("Received new thought from structured completion.")
                new_thought = chat_message.parsed
            else:
                self.logger.warning("No new thought received.")
                new_thought = None

            return new_thought

        except Exception as e:
            self.log_exception(e)
            self.logger.exception("Error occurred during step execution")
            await self.emit("error", self)
            return None

    @retry(stop=stop_after_attempt(5), reraise=True)
    async def run(self, messages: list[ChatCompletionMessageParam] = [], max_steps: int = DEFAULT_MAX_STEPS) -> Optional[Any]:
        """
        Runs the agent's reasoning loop to process tasks.

        Args:
            max_steps (int): Maximum number of reasoning steps to prevent infinite loops.
            **kwargs: Additional keyword arguments for initialization.

        Returns:
            Optional[Any]: The final result of the agent's reasoning process or None if terminated early.
        """
        self.logger.info("Agent run started.")
        self.max_steps = max_steps
        try:
            self.thought = await self.initialize(messages=messages)
            self.step_count = 0

            while self.thought and self.step_count < max_steps:
                self.logger.debug(f"Step {self.step_count + 1}: {self.thought}")
                self.thought = await self.step(self.thought)

                if self.step_count == max_steps // 2:
                    self.add_message({"role": "system", "content": "You are halfway before you will be forced to finish your task."})

                self.step_count += 1
                self.logger.debug(f"Completed step {self.step_count}.")

            if self.step_count >= max_steps:
                await self.emit("max_steps_reached", self)
                self.logger.warning(f"Maximum steps reached ({max_steps}). Terminating the run.")
            else:
                self.logger.info("Agent run completed successfully.")

            await self.emit("complete", self)
            return None  # Replace with actual result if available

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            self.logger.error("Agent run terminated due to an unexpected error.")
            await self.emit("error", self)
            raise e

    async def act(self) -> Optional[Any]:
        """
        Executes the actions specified in the tool calls.

        Returns:
            Optional[Any]: The result of the executed actions or None if no actions were performed.
        """
        self.logger.info("Agent act started.")
        await self.emit("before_act", self)
        try:
            messages = self.memory.get()  # Assuming `get` retrieves all messages
            await self.call_tool(messages=messages, parallel_tool_calls=self.parallel_tool_calls)
            self.logger.info("Agent act completed successfully.")
            await self.emit("after_act", self)
            return None

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            self.logger.error("Agent act terminated due to an unexpected error.")
            await self.emit("act_error")
            return None

    async def _reset(self):
        self.thought = None
        self.steps_count = 0
        self.has_initialized = None
        await self.emit("reset", self)
