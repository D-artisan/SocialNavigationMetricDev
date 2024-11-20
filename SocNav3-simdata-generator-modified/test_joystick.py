import pygame
import time

def main():
    # Initialize Pygame and the joystick module
    pygame.init()
    pygame.joystick.init()

    # Check if any joysticks are connected
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No joystick detected. Please connect a joystick and try again.")
        return
    else:
        # Initialize the first joystick
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick detected: {joystick.get_name()}")
        axes = joystick.get_numaxes()
        print(f"Number of axes: {axes}")
        buttons = joystick.get_numbuttons()
        print(f"Number of buttons: {buttons}")
        hats = joystick.get_numhats()
        print(f"Number of hats (D-pads): {hats}")

    try:
        print("\nMove the joystick axes and press buttons to see their values.")
        print("Press Ctrl+C to exit.\n")
        while True:
            # Process event queue
            pygame.event.pump()

            # Read axis values
            axis_values = []
            for i in range(axes):
                axis = joystick.get_axis(i)
                axis_values.append(round(axis, 3))  # Rounded for readability

            # Read button states
            button_values = []
            for i in range(buttons):
                button = joystick.get_button(i)
                button_values.append(button)

            # Read hat positions
            hat_values = []
            for i in range(hats):
                hat = joystick.get_hat(i)
                hat_values.append(hat)

            # Print the values
            print(f"Axes: {axis_values} | Buttons: {button_values} | Hats: {hat_values}")

            # Wait a short period to avoid flooding the console
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting joystick test.")
    finally:
        # Clean up
        pygame.quit()

if __name__ == "__main__":
    main()
