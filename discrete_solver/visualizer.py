import xml.etree.ElementTree as ET

def to_dot(manager, P):
    """
    Generates a premium Graphviz DOT representation of the ZBDD.
    Dashed lines represent 0-branches (low), and solid lines represent 1-branches (high).
    """
    dot_lines = [
        "digraph ZBDD {",
        "  fontname=\"Helvetica,Arial,sans-serif\";",
        "  node [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10];",
        "  edge [fontname=\"Helvetica,Arial,sans-serif\", fontsize=8];",
        "  # Terminals",
        "  0 [shape=box, label=\"0\", style=filled, fillcolor=\"#f8d7da\", color=\"#dc3545\", width=0.3, height=0.3];",
        "  1 [shape=box, label=\"1\", style=filled, fillcolor=\"#d4edda\", color=\"#28a745\", width=0.3, height=0.3];"
    ]
    
    visited = set([0, 1])
    def traverse(node):
        if node in visited:
            return
        visited.add(node)
        
        var, low, high = manager.nodes[node]
        # Label node
        dot_lines.append(f"  {node} [shape=circle, label=\"{var}\", style=filled, fillcolor=\"#e2e3e5\", color=\"#6c757d\"];")
        
        # Edges
        # Low branch (dashed line)
        dot_lines.append(f"  {node} -> {low} [style=dashed, color=\"#dc3545\", label=\"0\"];")
        # High branch (solid line)
        dot_lines.append(f"  {node} -> {high} [style=solid, color=\"#007bff\", penwidth=1.5, label=\"1\"];")
        
        traverse(low)
        traverse(high)
        
    traverse(P)
    dot_lines.append("}")
    return "\n".join(dot_lines)

def to_ascii(manager, P):
    """
    Generates a readable ASCII console layout representing the ZBDD.
    Shows the hierarchical decomposition of the set family.
    """
    if P == 0:
        return "∅ (Terminal 0)"
    if P == 1:
        return "{∅} (Terminal 1)"
        
    visited = {}
    lines = []
    
    def print_node(node, prefix="", is_high=None):
        if node == 0:
            lines.append(f"{prefix}└── [0] (∅)")
            return
        if node == 1:
            lines.append(f"{prefix}└── [1] ({{∅}})")
            return
            
        var, low, high = manager.nodes[node]
        branch_lbl = ""
        if is_high is True:
            branch_lbl = "high (1) ─> "
        elif is_high is False:
            branch_lbl = "low (0) ──> "
            
        if node in visited:
            lines.append(f"{prefix}└── {branch_lbl}Ref Node_{node} ({var})")
            return
            
        visited[node] = True
        lines.append(f"{prefix}└── {branch_lbl}Node_{node} [var: {var}]")
        
        new_prefix = prefix + "    "
        lines.append(f"{new_prefix}├── Low (0):")
        print_node(low, new_prefix + "│   ", is_high=False)
        lines.append(f"{new_prefix}└── High (1):")
        print_node(high, new_prefix + "    ", is_high=True)

    print_node(P)
    return "\n".join(lines)

def to_svg(manager, P, filename=None, width=800, height=600):
    """
    Generates a standalone, beautiful SVG file representing the ZBDD.
    Draws nodes in horizontal layers according to variable ordering.
    """
    # 1. Collect all reachable nodes and group by variable index
    nodes_by_var = {} # var_index -> list of node_ids
    visited = set()
    
    def collect(node):
        if node in visited:
            return
        visited.add(node)
        
        var_idx = manager.var_index(node)
        if var_idx not in nodes_by_var:
            nodes_by_var[var_idx] = []
        nodes_by_var[var_idx].append(node)
        
        if node not in [0, 1]:
            _, low, high = manager.nodes[node]
            collect(low)
            collect(high)
            
    collect(P)
    
    # Sort layers
    sorted_layer_indices = sorted(list(nodes_by_var.keys()))
    
    # Coordinates mapping
    coords = {} # node_id -> (cx, cy)
    
    margin = 50
    layer_height = (height - 2 * margin) / max(len(sorted_layer_indices) - 1, 1)
    
    # Assign (cx, cy) layout
    for l_idx, var_idx in enumerate(sorted_layer_indices):
        layer_nodes = sorted(nodes_by_var[var_idx])
        cy = margin + l_idx * layer_height
        
        num_nodes = len(layer_nodes)
        spacing = (width - 2 * margin) / (num_nodes + 1)
        
        for n_idx, node in enumerate(layer_nodes):
            cx = margin + (n_idx + 1) * spacing
            coords[node] = (cx, cy)
            
    # Create SVG structure using ElementTree
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': str(width),
        'height': str(height),
        'viewBox': f"0 0 {width} {height}"
    })
    
    # Definitions for arrows
    defs = ET.SubElement(svg, 'defs')
    # Arrow for 1-edge (high)
    marker_high = ET.SubElement(defs, 'marker', {
        'id': 'arrow-high',
        'viewBox': '0 0 10 10',
        'refX': '22', # adjust for node radius offset
        'refY': '5',
        'markerWidth': '6',
        'markerHeight': '6',
        'orient': 'auto-start-reverse'
    })
    ET.SubElement(marker_high, 'path', {'d': 'M 0 0 L 10 5 L 0 10 z', 'fill': '#007bff'})
    
    # Arrow for 0-edge (low)
    marker_low = ET.SubElement(defs, 'marker', {
        'id': 'arrow-low',
        'viewBox': '0 0 10 10',
        'refX': '22',
        'refY': '5',
        'markerWidth': '6',
        'markerHeight': '6',
        'orient': 'auto-start-reverse'
    })
    ET.SubElement(marker_low, 'path', {'d': 'M 0 0 L 10 5 L 0 10 z', 'fill': '#dc3545'})
    
    # Background
    ET.SubElement(svg, 'rect', {
        'width': '100%',
        'height': '100%',
        'fill': '#f8f9fa'
    })
    
    # Draw Edges first (so they sit behind nodes)
    for node in visited:
        if node in [0, 1]:
            continue
        cx, cy = coords[node]
        _, low, high = manager.nodes[node]
        
        # 0-branch (low) -> dashed red
        lx, ly = coords[low]
        ET.SubElement(svg, 'line', {
            'x1': str(cx), 'y1': str(cy),
            'x2': str(lx), 'y2': str(ly),
            'stroke': '#dc3545',
            'stroke-width': '1.5',
            'stroke-dasharray': '5,5',
            'marker-end': 'url(#arrow-low)'
        })
        
        # 1-branch (high) -> solid blue
        hx, hy = coords[high]
        ET.SubElement(svg, 'line', {
            'x1': str(cx), 'y1': str(cy),
            'x2': str(hx), 'y2': str(hy),
            'stroke': '#007bff',
            'stroke-width': '2.0',
            'marker-end': 'url(#arrow-high)'
        })
        
    # Draw Nodes
    for node in visited:
        cx, cy = coords[node]
        if node == 0:
            # Red box for Terminal 0
            ET.SubElement(svg, 'rect', {
                'x': str(cx - 15), 'y': str(cy - 15),
                'width': '30', 'height': '30',
                'rx': '4', 'ry': '4',
                'fill': '#f8d7da', 'stroke': '#dc3545', 'stroke-width': '2'
            })
            txt = ET.SubElement(svg, 'text', {
                'x': str(cx), 'y': str(cy + 5),
                'text-anchor': 'middle', 'font-family': 'sans-serif',
                'font-size': '14px', 'font-weight': 'bold', 'fill': '#721c24'
            })
            txt.text = '0'
        elif node == 1:
            # Green box for Terminal 1
            ET.SubElement(svg, 'rect', {
                'x': str(cx - 15), 'y': str(cy - 15),
                'width': '30', 'height': '30',
                'rx': '4', 'ry': '4',
                'fill': '#d4edda', 'stroke': '#28a745', 'stroke-width': '2'
            })
            txt = ET.SubElement(svg, 'text', {
                'x': str(cx), 'y': str(cy + 5),
                'text-anchor': 'middle', 'font-family': 'sans-serif',
                'font-size': '14px', 'font-weight': 'bold', 'fill': '#155724'
            })
            txt.text = '1'
        else:
            # Grey circle for internal node
            var = manager.nodes[node][0]
            ET.SubElement(svg, 'circle', {
                'cx': str(cx), 'cy': str(cy), 'r': '16',
                'fill': '#e2e3e5', 'stroke': '#6c757d', 'stroke-width': '2'
            })
            txt = ET.SubElement(svg, 'text', {
                'x': str(cx), 'y': str(cy + 5),
                'text-anchor': 'middle', 'font-family': 'sans-serif',
                'font-size': '11px', 'font-weight': 'bold', 'fill': '#383d41'
            })
            txt.text = str(var)
            
    svg_data = ET.tostring(svg, encoding='utf-8', method='xml').decode('utf-8')
    # Add xml declaration
    svg_content = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_data
    
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
            
    return svg_content
