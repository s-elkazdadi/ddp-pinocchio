from time import sleep
import itertools
import pinocchio as pin
import numpy as np
import sys
import os
import ast
from os.path import expanduser, join

from pinocchio.visualize import GepettoVisualizer

pinocchio_model_dir = expanduser("~/pinocchio/models")
model_path = join(pinocchio_model_dir, "others/robots")

if "ur5" in sys.argv[1]:
    urdf_model_path = join(model_path, "ur_description/urdf/ur5_gripper.urdf")
    last = 56
    step = 10
    camera = [
        0.569916307926178,
        -5.239817142486572,
        3.2733800411224365,
        0.4985126554965973,
        0.04369298741221428,
        0.02992137148976326,
        0.8652634024620056,
    ]
    camera = [
        -0.6566745638847351,
        -1.880828619003296,
        1.5453341007232666,
        0.4996102750301361,
        -0.10747529566287994,
        -0.11606986820697784,
        0.8516844511032104,
    ]
    targets = [
        np.array([-0.253, 0.606, 0.597]),
        np.array([0.648, 0.109, 0.597]),
        np.array([0.0953, 0.109, 0.906]),
    ]
if "anymal" in sys.argv[1]:
    urdf_model_path = join(model_path, "anymal_b_simple_description/robots/anymal.urdf")
    last = 61
    step = 10
    camera = [
        -2.5742063522338867,
        4.453205108642578,
        3.535933494567871,
        -0.16592755913734436,
        0.4660158157348633,
        0.8483554124832153,
        -0.1886541098356247,
    ]

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, model_path, pin.JointModelFreeFlyer()
)
nq = model.nq - 7
q = pin.neutral(model)[7:]  # ignore base

viz = GepettoVisualizer(model, collision_model, visual_model)
viz.initViewer()
gui = viz.viewer.gui

gui.setCameraTransform(viz.windowID, camera)
gui.setBackgroundColor1(viz.windowID, [0, 1, 0, 1])
gui.setBackgroundColor2(viz.windowID, [0, 1, 0, 1])


def remove_light():
    for node in gui.getNodeList():
        node: str
        if "RemoveLightSources" in gui.getPropertyNames(node):
            gui.removeLightSources(node)


def make(q):
    return np.hstack([np.zeros((7,)), q])


def show(q=None):
    if type(q) == int and q == 0:
        q = np.zeros(nq)
    if q is None:
        q = np.pi * np.random.rand(nq)
        print(repr(q))
    if len(q) == nq:
        q = make(q)
    elif len(q) != nq + 7:
        q = make(np.zeros(nq))
    viz.display(q)


for node in gui.getNodeList():
    if node.startswith("world/pinocchio") or node.startswith("world/sphere"):
        gui.deleteNode(node, True)

if "ur5" in sys.argv[1]:
    for i in range(3):
        gui.addSphere("world/sphere" + str(i), 0.1, [1, 0, 0, 0.5])
        gui.applyConfiguration(
            "world/sphere" + str(i), list(np.hstack([targets[i], [1, 0, 0, 0]]))
        )
    gui.refresh()

viz.loadViewerModel("pinocchio")
remove_light()
show(0)

if False:
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        iterates = [ast.literal_eval(line.strip().rstrip(",")) for line in lines]
        for traj in iterates:
            for i in range(len(traj)):
                traj[i] = np.array(traj[i][:nq])

    if sys.argv[2] == "images":
        traj = iterates[-1]
        n = 50
        for i in range(n):
            show(traj[max(0, min(len(traj) - 1, int(i * len(traj) / (n - 1))))])
            gui.captureFrame(viz.windowID, f"ur5_{i:02}.png")
            sleep(2 / n)

    if sys.argv[2] == "iterates":
        count = 3
        for i in range(count):
            print(count - i)
            sleep(1)
        print(0)

        sleep_time = 0
        for i, traj in itertools.islice(enumerate(iterates), None, last, step):
            sleep(sleep_time)
            print(i)
            for k in range(3):
                gui.setColor("world/sphere" + str(k), [1, 0, 0, 0.5])
            found = [False, False, False, False]
            looking_for = [True, False, False, False]
            for j, x in enumerate(traj):
                pin.framesForwardKinematics(viz.model, viz.data, make(x))
                for k in range(3):
                    if found[k]:
                        continue
                    if (
                        looking_for[k]
                        and np.linalg.norm(targets[k] - viz.data.oMf[18].translation)
                        < 0.01
                    ):
                        gui.setColor("world/sphere" + str(k), [0, 1, 0, 0.5])
                        print(j)
                        found[k] = True
                        looking_for[k + 1] = True
                    else:
                        break
                show(x)
                sleep(0.01)
