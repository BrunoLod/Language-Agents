from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, trim_messages


class CleanMemory():

    def __init__(
            self, 
            max_messages: int = 10, 
            strategy: str = "last", 
            start_on: str = "human"
        ) -> None:
        
        self.max_messages = max_messages
        self.strategy     = strategy
        self.start_on     = start_on

    def check_len(self, memory) -> bool:
        """ 
        """
        return len(memory.messages) > self.max_messages
    
    def without_first_memory(self, memory) -> list:
        """ 
        """
        return memory.messages[1:]
    
    def get_first_memory(self, memory) -> list:
        """ 
        """
        return memory.messages[:1]
    
    def manage_memory(self, messages: list) -> list:
        """ 
        """
        return trim_messages(
            messages, 
            token_counter  = len, 
            max_tokens     = self.max_messages, 
            strategy       = self.strategy, 
            start_on       = self.start_on, 
            include_system = False
        )
    
    def trim_messages(self, memory) -> None:
        """ 
        """
        if self.check_len(memory=memory):

            list_messages = self.without_first_memory(memory=memory)
            first_message = self.get_first_memory(memory=memory)

            trimmed_messages = first_message + self.manage_memory(list_messages)

        else:
            trimmed_messages = memory.messages

        # Clear the current memory and update it with trimmed message
        memory.clear()
        for message in trimmed_messages:
            memory.add_message(message)