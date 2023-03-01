def preprocessing(frame):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    frame = frame[35:195] # crop
    frame = frame[::2, ::2, 0] # downsample by factor of 2
    frame[frame == 144] = 0 # erase background (background type 1)
    frame[frame == 109] = 0 # erase background (background type 2)
    frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
    return frame.astype(float).ravel()
