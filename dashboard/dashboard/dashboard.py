"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx

from rxconfig import config


class State(rx.State):
    """The app state."""

    ...


def glm_results(category):
    return rx.vstack(
        rx.hstack(
            rx.image(
                src=f"out/analysis_weekly_workload/{category}/coef.png",
                width="400px",
                height="auto",
            ),
            rx.hstack(
                rx.image(
                    src=f"out/analysis_weekly_workload/{category}/scatter_pred_true.png",
                    width="400px",
                    height="auto",
                ),
            )
        ),
    )

def glm_results_delta(category):
    return rx.vstack(
        rx.hstack(
            rx.image(
                src=f"out/analysis_weekly_workload/{category}/coef.png",
                width="400px",
                height="auto",
            ),
            rx.hstack(
                rx.image(
                    src=f"out/analysis_weekly_workload/{category}/scatter_pred_true.png",
                    width="400px",
                    height="auto",
                ),
            )
        ),
        rx.hstack(
            rx.image(
                src=f"out/analysis_weekly_workload/{category}_delta/coef.png",
                width="400px",
                height="auto",
            ),
            rx.hstack(
                rx.image(
                    src=f"out/analysis_weekly_workload/{category}_delta/scatter_pred_true.png",
                    width="400px",
                    height="auto",
                ),
            )
        ),
    )


def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Athlete Workload and Wellness Report", size="9"),
            rx.text("Perry Battles, Hal H. Morris Human Performance Laboratories, Indiana University"),
            rx.text(
                "Code available at ",
                rx.link(
                    rx.code("https://github.com/Battles186/USOPC_interview_project.git"),
                ),
                size="5",
                href="https://github.com/Battles186/USOPC_interview_project.git",
            ),
            # rx.link(
            #     "Code available at ",
            #     rx.code("https://github.com/Battles186/USOPC_interview_project.git"),
            #     size="5",
            #     href="https://github.com/Battles186/USOPC_interview_project.git",
            # ),

            rx.heading("EDA", size="7"),
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("Histograms", value="0_histograms"),
                    rx.tabs.trigger("By Player Group", value="1_by_player_group"),
                ),
                rx.tabs.content(
                    rx.image(
                        src="images/eda/eda_hist_all.png",
                        width="800px",
                        height="auto",
                    ),
                    value="0_histograms",
                ),
                rx.tabs.content(
                    rx.image(
                        src="images/eda/eda_box_all_pos_group.png",
                        width="800px",
                        height="auto",
                    ),
                    value="1_by_player_group",
                ),
            ),

            rx.heading("Generalized Linear Modeling", size="7"),
            rx.text("Model Fits"),
            rx.tabs.root(
                rx.tabs.list(
                    rx.tabs.trigger("Fatigue", value="0_fatigue"),
                    rx.tabs.trigger("Mood", value="1_mood"),
                    rx.tabs.trigger("Motivation", value="2_motivation"),
                    rx.tabs.trigger("Stress", value="3_stress"),
                    rx.tabs.trigger("Sleep Disturbance", value="4_sleep_disturbance"),
                    rx.tabs.trigger("Soreness", value="5_workload_distance_sleep_soreness"),
                ),
                rx.tabs.content(
                    glm_results_delta('workload_sleep_fatigue'),
                    value="0_fatigue",
                ),
                rx.tabs.content(
                    glm_results_delta('workload_sleep_mood'),
                    value="1_mood",
                ),
                rx.tabs.content(
                    glm_results_delta('workload_sleep_motivation'),
                    value="2_motivation",
                ),
                rx.tabs.content(
                    glm_results_delta('workload_sleep_stress'),
                    value="3_stress",
                ),
                rx.tabs.content(
                    glm_results('workload_sleep'),
                    value="4_sleep_disturbance",
                ),
                rx.tabs.content(
                    glm_results('workload_distance_sleep_soreness'),
                    value="5_workload_distance_sleep_soreness",
                ),
            ),

            rx.heading("Conclusions", size="7"),
            rx.accordion.root(
                rx.accordion.item(
                    header="EDA",
                    content="EDA demonstrated that average session heart rate and practice load likely express much of the same information. If collecting both is logistically intractable or difficult, not much is lost by sacrificing one. Because of the familiarity with heart rate data and zones, it may be more useful to present heart rate data to coaches and athletes."
                ),
                rx.accordion.item(
                    header="Sleep and athlete wellness",
                    content="Sleep has by far the most outsized effect in mitigating athlete wellness by reducing fatigue and stress and increasing mood and motivation.",
                ),
                rx.accordion.item(
                    header="Workload and sleep quality",
                    content="Workload does not appear to affect sleep quality within the ranges in which it has been administered, with the exception of a potential slight effect for practice load.",
                ),
                rx.accordion.item(
                    header="Soreness",
                    content="Soreness was negatively associated with sleep to a much greater degree than with any other variable. Competition load did positively influence soreness.",
                ),
                rx.accordion.item(
                    header="Strength and Conditioning",
                    content="Athletes that performed more strength and conditioning in a given week experienced less stress and fatigue. However, these effects are likely very slight and possibly confounded by the positions of players engaging in the most strength and conditioning also performing less other training.",
                ),
                width="500px",
            ),

            spacing="5",
            justify="center",
            min_height="85vh",
        ),
        rx.logo(),
    )


app = rx.App()
app.add_page(index)
