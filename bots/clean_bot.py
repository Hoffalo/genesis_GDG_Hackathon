import random


class MyBot():
    def __init__(self):
        self.name = "MyBot"

    # IMPLEMENT YOUR METHODS HERE

    # Modify but don't rename this method
    def act(self, info):
        # Receives info dictionary from the game (for the proper player)
        """
        info = {
            "location": self.get_location(),
            "rotation": self.get_rotation(),
            "rays": self.get_rays(),
            "current_ammo": self.current_ammo
        }
        """

        # Should return a dictionary of moves, for example:
        actions = {
            "forward": True,
            "right": False,
            "down": False,
            "left": False,
            "rotate": 0,
            "shoot": True
            } # This will make the bot go forward and shoot
        # Include even non used variables

        direction = random.choice(["forward", "right", "down", "left"])
        actions = {
            "forward": direction == "forward",
            "right": direction == "right",
            "down": direction == "down",
            "left": direction == "left",
            "rotate": random.randint(-180, 180),  # Rotate randomly
            "shoot": random.choice([True, False])  # Randomly decide to shoot or not
        }

        return actions