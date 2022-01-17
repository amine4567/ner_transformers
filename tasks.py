from invoke import task


@task
def update_app_reqs(c):
    c.run("pip-compile src/requirements.in")


@task
def update_dev_reqs(c):
    c.run("pip-compile requirements-dev.in")
