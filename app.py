import cv2
import numpy as np
import random
import mediapipe as mp

# Constants
WIDTH, HEIGHT = 800, 600
SNAKE_RADIUS = 10
FOOD_RADIUS = 15
SPEED = 10

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

class SnakeGame:
    def __init__(self):
        self.snake = [[100, 100]]  # Snake starting position
        self.food = [random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)]
        self.score = 0
        self.running = True
        self.grow_snake = False

    def move_snake(self, finger_pos):
        """Move the snake's head towards the fingertip."""
        if finger_pos:
            x, y = finger_pos
            # Ensure the new head follows the finger smoothly
            new_head = [x, y]

            # Add new head to the snake
            self.snake.insert(0, new_head)

            # Check for food collision
            if self.check_food_collision():
                self.score += 1
                self.grow_snake = True
                self.spawn_food()
            else:
                self.grow_snake = False

            # Remove tail unless growing
            if not self.grow_snake:
                self.snake.pop()

    def spawn_food(self):
        """Randomly spawn new food ensuring it doesn't overlap the snake."""
        while True:
            new_food = [random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)]
            if all(np.linalg.norm(np.array(new_food) - np.array(segment)) > FOOD_RADIUS + SNAKE_RADIUS for segment in self.snake):
                self.food = new_food
                break

    def check_food_collision(self):
        """Check if the snake's head collides with the food."""
        head = self.snake[0]
        return np.linalg.norm(np.array(head) - np.array(self.food)) < (FOOD_RADIUS + SNAKE_RADIUS)

    def check_collision(self):
        """Check for collisions with the snake itself or boundaries."""
        head = self.snake[0]
        # Boundary collision
        if head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT:
            return True
        # Self-collision
        for segment in self.snake[1:]:
            if np.linalg.norm(np.array(head) - np.array(segment)) < SNAKE_RADIUS * 2:
                return True
        return False

    def draw(self, frame):
        """Draw the snake and food on the frame."""
        # Draw snake as circles
        for segment in self.snake:
            cv2.circle(frame, tuple(segment), SNAKE_RADIUS, (0, 255, 0), -1)
        # Draw food as a circle
        cv2.circle(frame, tuple(self.food), FOOD_RADIUS, (0, 0, 255), -1)
        # Display score
        cv2.putText(frame, f"Score: {self.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    game = SnakeGame()

    while game.running:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and resize frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Convert to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        finger_pos = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_pos = [int(index_finger_tip.x * WIDTH), int(index_finger_tip.y * HEIGHT)]

        # Update snake's position
        if finger_pos:
            game.move_snake(finger_pos)

        # Check for collisions
        if game.check_collision():
            game.running = False

        # Draw game elements
        game.draw(frame)

        # Show frame
        cv2.imshow("Snake Game", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Game Over! Your score: {game.score}")


if __name__ == "__main__":
    main()
