import os
from pathlib import Path

import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import (
    # xml_path_completion,
    find_elements,
    array_to_string,
)

# Get project root directory (3 levels up from this file)
# This file is at: experiments/robosuite/xml_objects.py
# Project root is at: LoopAct/
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
PATH_TO_OBJECT_MODELS = str(_project_root / "experiments" / "models" / "objects")

def xml_path_completion(xml_path, root=None):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path
        root (str): root location of assets (only applicable if path is not absolute)

    Returns:
        str: Full (absolute) xml path
    """
    if xml_path.startswith("/"):
        full_path = xml_path
    else:
        if root is None:
            root = PATH_TO_OBJECT_MODELS
        full_path = os.path.join(root, xml_path)
    return full_path


class PrefixedMujocoXMLObject(MujocoXMLObject):
    """
    XML objects whose geoms already include a name prefix in the source XML.
    Avoids double-prefixing in correct_naming while still adding prefixes where needed.
    """

    def exclude_from_prefixing(self, inp):
        if isinstance(inp, str):
            return inp.startswith(self.naming_prefix)
        if hasattr(inp, "get"):
            name = inp.get("name")
            return bool(name and name.startswith(self.naming_prefix))
        return False


class PlateObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("dinnerware/plate.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

class TableObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("table.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class PotObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("pot.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class PotLidObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("pot_lid.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class UtensilCaseObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("utensil_case.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class LadleObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("ladle.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class StoveObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("stove.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ShelfObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("shelf.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic


class CabinetObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("cabinet.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic


class NailObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("nail.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic


class PegboardObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("pegboard.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[
                dict(
                    name="joint_x",
                    type="slide",
                    axis="1 0 0",
                    frictionloss="100",
                    damping="0.1",
                ),
                dict(
                    name="joint_y",
                    type="slide",
                    axis="0 1 0",
                    frictionloss="100",
                    damping="0.1",
                ),
                dict(
                    name="joint_yaw",
                    type="hinge",
                    axis="0 0 1",
                    frictionloss="100",
                    damping="0.1",
                ),
            ],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class HandObject(PrefixedMujocoXMLObject):
    """Inspire right hand model (rigid, no finger joints) for use as a floating visual object."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("hand/hand.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ToyCartObject(PrefixedMujocoXMLObject):
    """Flat toy cart built from MuJoCo primitives (tabletop scale, ~18cm long)."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("toy_cart.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ToyRobotObject(PrefixedMujocoXMLObject):
    """Toy robot built from MuJoCo primitives (tabletop scale, ~10cm tall)."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("toy_robot.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class BlockOnCartObject(PrefixedMujocoXMLObject):
    """Blue block sitting on a cart — single rigid body.
    Same collision footprint as ToyRobotRedObject; total height above table surface equals toy robot."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("block_on_cart.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ToyRobotRedObject(PrefixedMujocoXMLObject):
    """Red toy robot — same shape as ToyRobotObject but with a red colour scheme."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("toy_robot_red.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ToyRobotGreenObject(PrefixedMujocoXMLObject):
    """Green toy robot — same shape/physics as ToyRobotRedObject but with a green colour scheme."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("toy_robot_green.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class SodaCanObject(PrefixedMujocoXMLObject):
    """Soda can built from MuJoCo cylinder primitives."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("soda_can.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ConeObject(PrefixedMujocoXMLObject):
    """Tapered cone built from a stack of MuJoCo cylinder slices."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("cone.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class CupObject(PrefixedMujocoXMLObject):
    """Open-top square cup, ~8cm outer / 7cm inner. Larger than a 5cm ball."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("cup.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class OrangeCupObject(PrefixedMujocoXMLObject):
    """Open-top square cup — same geometry as CupObject but orange. Used by drop_ball2."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("cup_orange.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class BigCartObject(PrefixedMujocoXMLObject):
    """Big-wheel cart: platform sits at wheel equator, invisible pedestal for physics."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("big_cart.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class HighBlockRedObject(PrefixedMujocoXMLObject):
    """Tall red block with invisible bottom 2.6cm — same collision as toy robot, visual hovers at cart height."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("high_block_red.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class HighBlockCartObject(PrefixedMujocoXMLObject):
    """Visual-only cart for HighBlock — tiny physical anchor at +y gives body mass; rest is display only."""

    def __init__(self, name):
        super().__init__(
            xml_path_completion("high_block_cart.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[dict(type="free")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class CartObject(PrefixedMujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        super().__init__(
            xml_path_completion("cart.xml", PATH_TO_OBJECT_MODELS),
            name=name,
            joints=[
                dict(
                    name="joint_x",
                    type="slide",
                    axis="1 0 0",
                    frictionloss="0.5",
                    damping="320.",
                ),
                dict(
                    name="joint_y",
                    type="slide",
                    axis="0 1 0",
                    frictionloss="0.5",
                    damping="320.",
                ),
                dict(
                    name="joint_yaw",
                    type="hinge",
                    axis="0 0 1",
                    frictionloss="0.5",
                    damping="640.",
                ),
            ],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
