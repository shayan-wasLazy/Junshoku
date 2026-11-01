from .nodes import masked_node, exposure_wb, tonal_contrast, creative_colour

def apply_graph(source, output, cfg):
    p=output["params"]; maps=output["maps"]
    x=masked_node(source,exposure_wb(source,p[0],cfg),maps[0])
    x=masked_node(x,tonal_contrast(x,p[1],cfg),maps[1])
    x=masked_node(x,creative_colour(x,p[2],cfg),maps[2])
    return x
