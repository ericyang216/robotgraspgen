from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

class BreadObject(MujocoXMLObject):
    def __init__(self):
        super().__init__(xml_path_completion("objects/bread.xml"))
