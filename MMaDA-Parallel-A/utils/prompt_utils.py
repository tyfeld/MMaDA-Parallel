# -*- coding: utf-8 -*-
"""
Prompt generation utilities for different inference types
"""
from typing import Dict, List, Tuple, Optional

def create_prompt_templates():
    """Create prompt templates for various tasks"""
    templates = {
        "text_understanding": "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer.",
        "image_generation": "Generate an image according to the text prompt.",
        "image_editing": "Generate an image applying the following editing instruction based on the original image.",
        "dense_prediction": "Perform dense prediction on the given images.",
        "control_generation": "Generate an image according to the text prompt and the given control image.",
        "subject_generation": "Generate an image according to the text prompt and the given object image.",
        "multi_view": "Generate a view-image based on the given image.",
        "style_transfer": "Transform the current image into the style of the provided image."
    }
    return templates


def generate_text_to_image_prompt(prompt_text: str, templates: Optional[Dict] = None) -> Tuple[str, str]:
    """
    Generate prompt for text-to-image generation
    
    Args:
        prompt_text: User input text prompt
        templates: Optional prompt templates dict
        
    Returns:
        Tuple of (input_prompt, unconditional_prompt)
    """
    if templates is None:
        templates = create_prompt_templates()
    
    system_prompt = templates["image_generation"]
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + prompt_text + "</user>"
    uncon_prompt = "<system>" + system_prompt + "</system>" + "<user>" + "<uncondition>" + "</user>"
    
    return input_prompt, uncon_prompt


def generate_image_to_image_prompt(
    prompt_text: str, 
    edit_type: str, 
    templates: Optional[Dict] = None,
    **kwargs
) -> Tuple[str, str, str]:
    """
    Generate prompt for image-to-image generation
    
    Args:
        prompt_text: User input text prompt
        edit_type: Type of editing operation
        templates: Optional prompt templates dict
        **kwargs: Additional parameters for specific edit types
        
    Returns:
        Tuple of (input_prompt, unconditional_prompt, system_prompt)
    """
    if templates is None:
        templates = create_prompt_templates()
    
    # Determine system prompt and processed prompt text based on edit type
    if 'dense' in edit_type:
        des = {
            "canny": "canny edge map", 
            "hed": "hed edge map", 
            "normal": "normal map",
            "sam2mask": "sam2 mask", 
            "depth": "depth map", 
            "openpose": "pose estimation map"
        }
        system_prompt = templates["dense_prediction"]
        prompt_text_used = f"Generate a {des.get(edit_type.split('_')[0], 'dense map')} according to the image."
        
    elif 'control' in edit_type:
        system_prompt = templates["control_generation"]
        prompt_text_used = prompt_text
        
    elif 'subject' in edit_type:
        system_prompt = templates["subject_generation"]
        prompt_text_used = prompt_text
        
    elif 'edit' in edit_type:
        system_prompt = templates["image_editing"]
        prompt_text_used = prompt_text
            
    elif "ref_transfer" in edit_type:
        system_prompt = templates["style_transfer"]
        prompt_text_used = "Transform the current image into the style of the provided image."
        
    elif 'multi_view' in edit_type:
        system_prompt = templates["multi_view"]
        prompt_text_used = f"Generate the {edit_type.split('_')[-1]} view based on the provided front view."
        
    else:
        system_prompt = "Generate an image according to the prompt and image."
        prompt_text_used = prompt_text
    
    # Build final prompts
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + prompt_text_used + "</user>"
    uncon_prompt = "<system>" + system_prompt + "</system>" + "<user>" + "<uncondition>" + "</user>"
    
    return input_prompt, uncon_prompt, system_prompt


def generate_multimodal_understanding_prompt(question: str, templates: Optional[Dict] = None) -> str:
    """
    Generate prompt for multimodal understanding (MMU)
    
    Args:
        question: User question about the image
        templates: Optional prompt templates dict
        
    Returns:
        Formatted input prompt
    """
    if templates is None:
        templates = create_prompt_templates()
    
    system_prompt = "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer."
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + question + "</user>"
    
    return input_prompt


def get_edit_type_specific_prompt(edit_type: str, prompt_text: str, templates: Optional[Dict] = None) -> str:
    """
    Get edit type specific prompt text
    
    Args:
        edit_type: Type of editing operation
        prompt_text: Original prompt text
        templates: Optional prompt templates dict
        
    Returns:
        Processed prompt text for the specific edit type
    """
    if templates is None:
        templates = create_prompt_templates()
    
    if 'dense' in edit_type:
        des = {
            "canny": "canny edge map", 
            "hed": "hed edge map", 
            "normal": "normal map",
            "sam2mask": "sam2 mask", 
            "depth": "depth map", 
            "openpose": "pose estimation map"
        }
        return f"Generate a {des.get(edit_type.split('_')[0], 'dense map')} according to the image."
        
    elif 'control' in edit_type:
        return prompt_text
        
    elif 'subject' in edit_type:
        return prompt_text
        
    elif 'edit' in edit_type:
        if "multiturn" in edit_type:
            ids = int(edit_type.split("_")[-1])
            if ids == 0:
                return prompt_text[0] if isinstance(prompt_text, list) else prompt_text
            else:
                return prompt_text[ids][0] if isinstance(prompt_text[ids], list) else prompt_text[ids]
        else:
            return prompt_text
            
    elif "ref_transfer" in edit_type:
        return "Transform the current image into the style of the provided image."
        
    elif 'multi_view' in edit_type:
        return f"Generate the {edit_type.split('_')[-1]} view based on the provided front view."
        
    else:
        return prompt_text


def get_system_prompt_for_edit_type(edit_type: str, templates: Optional[Dict] = None) -> str:
    """
    Get system prompt for specific edit type
    
    Args:
        edit_type: Type of editing operation
        templates: Optional prompt templates dict
        
    Returns:
        System prompt for the edit type
    """
    if templates is None:
        templates = create_prompt_templates()
    
    if 'dense' in edit_type:
        return templates["dense_prediction"]
    elif 'control' in edit_type:
        return templates["control_generation"]
    elif 'subject' in edit_type:
        return templates["subject_generation"]
    elif 'edit' in edit_type:
        return templates["image_editing"]
    elif "ref_transfer" in edit_type:
        return templates["style_transfer"]
    elif 'multi_view' in edit_type:
        return templates["multi_view"]
    else:
        return "Generate an image according to the prompt and image."

def generate_text_image_to_text_image_prompt(prompt_text, system_prompt):
    """
    Generate prompts for TI2TI tasks
    
    Args:
        prompt_text: User's editing instruction
        system_prompt: System prompt for the task
    
    Returns:
        input_prompt: Conditional prompt
        uncon_text: Unconditional prompt
    """
    # Conditional prompt
    input_prompt = (
        f"<system>{system_prompt}</system>"
        f"<user>{prompt_text}</user>"
    )
    
    # Unconditional prompt (for CFG)
    uncon_text = (
        f"<system>{system_prompt}</system>"
        f"<user><uncondition></user>"
    )
    
    return input_prompt, uncon_text