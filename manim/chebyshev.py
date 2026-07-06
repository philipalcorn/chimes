from manim import *
import numpy as np

class ChebyshevAnimation(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-1, 1, 0.5],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=5,
            tips=False,
        )

        self.add(axes)

        n = ValueTracker(0)

        graph = always_redraw(
            lambda: axes.plot(
                lambda x: np.cos(
                    n.get_value() * np.arccos(np.clip(x, -1, 1))
                ),
                x_range=[-1, 1],
                color=BLUE,
            )
        )

        label = MathTex(r"T_0(x)").to_edge(UP)
        self.add(label)

        self.add(graph, label)
        for k in range(8):
            # Pause at the current integer
            self.wait(2)

            new_label = MathTex(rf"T_{{{k+1}}}(x)").to_edge(UP)


            # Quickly transition to the next integer
            self.play(
                n.animate.set_value(k + 1),
                Transform(label, new_label),
                run_time=0.5,
                rate_func=smooth,
            )


        self.wait(1)
