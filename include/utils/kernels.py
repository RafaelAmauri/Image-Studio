import numpy as np

boxBlur3x3 = np.asarray(
                        [
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]
                        ])

boxBlur5x5 = np.asarray(
                        [
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]
                        ])

gaussianBlur3x3 = np.asarray(
                            [
                                [1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]
                            ])

gaussianBlur5x5 = np.asarray(
                            [
                                [1,  4,  7,  4, 1],
                                [4, 16, 26, 16, 4],
                                [7, 26, 41, 26, 7],
                                [4, 16, 25, 16, 4],
                                [1,  4,  7,  4, 1]
                            ])

sobelHorizontal3x3 = np.asarray(
                            [
                                [ 1,  2,  1],
                                [ 0,  0,  0],
                                [-1, -2, -1]
                            ])

sobelVertical3x3 = np.asarray(
                            [
                                [1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]
                            ]
)

prewittHorizontal3x3 = np.asarray(
                            [
                                [ 1,  1,  1],
                                [ 0,  0,  0],
                                [-1, -1, -1]
                            ])


prewittVertical3x3 = np.asarray(
                            [
                                [ 1, 0, -1],
                                [ 1, 0, -1],
                                [ 1, 0, -1]
                            ])