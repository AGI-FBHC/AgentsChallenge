from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
# Assuming SimulationAgent and interaction_tool are already defined in your environment
from websocietysimulator import Simulator

class MySimulationAgent(SimulationAgent):
    def __init__(self, task):
        # Initialize the agent with the task
        self.task = task

    def workflow(self):
        # Extract user_id and item_id from the task input
        user_id = self.task.get("user_id")
        item_id = self.task.get("item_id")

        # The simulator will automatically set the task for your agent. You can access the task by `self.task` to get task information.
        print(f"Task received: {self.task}")

        # Use the interaction_tool to get data from the dataset using user_id and item_id
        item_info = self.interaction_tool.get_item(item_id=item_id)
        user_info = self.interaction_tool.get_user(user_id=user_id)
        item_re = self.interaction_tool.get_reviews(review_id=item_id)  # Using item_id to fetch reviews for this item
        user_re = self.interaction_tool.get_reviews(review_id=user_id)  # Assuming reviews are linked with user_id

        # Print out the information for debugging
        print(f"User Info: {user_info}")
        print(f"User Reviews: {user_re}")
        print(f"Item Info: {item_info}")
        print(f"Item Reviews: {item_re}")

        # Implement your logic here to process data and generate review and rating
        # You can perform any necessary data processing or decision-making based on the data you retrieved

        # Finally, you need to return the result in the format of `stars` and `review`.
        stars = 4.0
        review = "Great experience!"
        return stars, review


# Main function where you input the task
def main():
    # Define the task as input
    task = {
        "type": "user_behavior_simulation",
        "user_id": "AF6HG5ZKZR4N2BTWJ7FGMXZO3JEQ",
        "item_id": "B09LLZN3V3"
    }

    # Instantiate the agent with the task
    my_agent = MySimulationAgent(task)

    # Run the workflow method which simulates the task processing
    stars, review = my_agent.workflow()

    # Output the results
    print(f"Stars: {stars}")
    print(f"Review: {review}")


# Call the main function to run the simulation
if __name__ == "__main__":
    main()
