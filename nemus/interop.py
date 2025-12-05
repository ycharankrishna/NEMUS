import json
import numpy as np
from typing import Any
import importlib

class NIR_Bridge:
    def __init__(self):
        pass

    def export(self, network: Any, format: str = "nir-x") -> str:
        graph_data = {
            "layers": [],
            "connections": []
        }
        
        for i, layer in enumerate(network.layers):
            layer_data = {
                "name": layer.name,
                "class": layer.__class__.__name__,
                "module": layer.__class__.__module__,
                "params": {}
            }
            # Extract basic params
            if hasattr(layer, 'neurons'): layer_data["params"]["neurons"] = layer.neurons
            if hasattr(layer, 'shape'): layer_data["params"]["shape"] = layer.shape
            
            # Extract Model params if Recurrent
            if hasattr(layer, 'model'):
                layer_data["params"]["model"] = {
                    "class": layer.model.__class__.__name__,
                    "params": {k: v for k, v in layer.model.__dict__.items() if isinstance(v, (int, float, str))}
                }
            
            graph_data["layers"].append(layer_data)
            
            if layer.next_layer:
                graph_data["connections"].append({
                    "from": layer.name,
                    "to": layer.next_layer.name
                })
                
        return json.dumps(graph_data, indent=2)

    def load(self, path: str) -> Any:
        # Reconstruct Network from JSON
        with open(path, 'r') as f:
            data = json.load(f)
            
        layers_map = {}
        ordered_layers = []
        
        # 1. Create Layers
        for l_data in data["layers"]:
            module_name = l_data["module"]
            class_name = l_data["class"]
            params = l_data["params"]
            
            # Dynamic import
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            
            # Handle special cases (Recurrent needs model)
            if "model" in params:
                model_data = params.pop("model")
                # Assume model is in nemus.biology
                bio_module = importlib.import_module("nemus.biology")
                model_cls = getattr(bio_module, model_data["class"])
                model_instance = model_cls(**model_data["params"])
                params["model"] = model_instance
                
            # Instantiate
            # We need to filter params that match __init__? 
            # For simplicity, we assume params match.
            # Some params like 'shape' might need tuple conversion
            if "shape" in params: params["shape"] = tuple(params["shape"])
            
            layer = cls(**params)
            layer.name = l_data["name"] # Restore name
            layers_map[layer.name] = layer
            ordered_layers.append(layer)
            
        # 2. Connect
        for conn in data["connections"]:
            src = layers_map[conn["from"]]
            dst = layers_map[conn["to"]]
            src.connect(dst)
            
        # 3. Create Network
        # Assume first layer is entry
        from .network import Network
        if not ordered_layers: return None
        return Network(ordered_layers[0])
