from manim import *
import numpy as np


class ChebyshevAnimation(Scene):

    def chebyshev(self, n, x):
        return np.cos(n * np.arccos(np.clip(x, -1, 1)))

    def make_graph(self, axes, n):
        return axes.plot(
            lambda x: self.chebyshev(n, x),
            x_range=[-1, 1],
            color=BLUE,
        )

    def construct(self):
        axes = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=5,
            tips=False,
        )

        self.add(axes)

        graph = self.make_graph(axes, 0)

        label = MathTex(r"T_0(x)").to_edge(UP)

        self.add(graph, label)

        for k in range(8):
            self.wait(1)

            new_graph = self.make_graph(axes, k + 1)
            new_label = MathTex(rf"T_{{{k+1}}}(x)").to_edge(UP)

            self.play(
                Transform(graph, new_graph),
                Transform(label, new_label),
                run_time=0.5,
                rate_func=smooth,
            )

        self.wait(1)
