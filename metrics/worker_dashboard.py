def loss_time(vis, state):
    vis.line(state["loss"][-1:], [sum(state["compute_time"])], win="loss_time", 
        name=f"worker_{state['id']}", update="append",
        opts=dict(
                showlegend=True,
                xlabel="Time, s",
                ylabel="Loss"
            ),
    )

def loss_step(vis, state):
    vis.line(state["loss"][-1:], [state["step_num"]], win="loss_step", 
        name=f"worker_{state['id']}", update="append",
        opts=dict(
                showlegend=True,
                xlabel="Step",
                ylabel="Loss"
            ),
    )

def time_step(vis, state):
    vis.line(state["compute_time"][-1:], [state["step_num"]], win="time_step", 
        name=f"worker_{state['id']}", update="append",
        opts=dict(
                showlegend=True,
                xlabel="Step",
                ylabel="Time, s"
            ),
    )

def time_hist(vis, state):
    if len(state["compute_time"]) > 1:
        vis.histogram(state["compute_time"], win="time_hist")


metrics = [loss_time, loss_step, time_step, time_hist]