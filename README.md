Third version of the code for my lego sorting machine.
Uses:
- yolo model for detecting where bricks are
- cv2 to process the location of the bricks
- zones which trigger events when bricks enter and leave
- moves the sorter accordingly (via a rpi pico running the code from my Picostepper repo)
- recognizes the brick id for each brick and displays an image of said brick (using 2 APIs)
