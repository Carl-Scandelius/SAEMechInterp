# import torch
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sklearn.decomposition import PCA
# from tqdm import tqdm
# import gc
# import json


# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# TARGET_LAYER = 21
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

# -
# def get_model_and_tokenizer(model_name):
#     """Loads the model and tokenizer."""
#     print(f"Loading model: {model_name}...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=TORCH_DTYPE,
#         device_map=DEVICE
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     print("Model and tokenizer loaded.")
#     return model, tokenizer

# # --- ULTIMATELY CORRECTED HELPER FUNCTION ---
# def find_concept_token_index(tokenizer, prompt_text, concept_word):
#     """
#     Finds the index of the last token corresponding to the concept_word in a prompt
#     using a robust method that relies on a single tokenization pathway.
#     """
#     # 1. Prepare two versions of the prompt for a differential analysis.
#     # The first prompt has the concept word, the second has a single, unique placeholder token.
#     placeholder = "[PLACEHOLDER]"
#     # It's crucial to add the placeholder as a special token so it's not split.
#     tokenizer.add_special_tokens({'additional_special_tokens': [placeholder]})
    
#     prompt_with_concept = prompt_text
#     prompt_with_placeholder = prompt_text.replace(concept_word, placeholder, 1) # Replace only the first instance

#     # 2. Tokenize both prompts using the exact same chat template.
#     messages_concept = [{"role": "user", "content": prompt_with_concept}]
#     messages_placeholder = [{"role": "user", "content": prompt_with_placeholder}]

#     ids_concept = tokenizer.apply_chat_template(messages_concept, add_generation_prompt=False)
#     ids_placeholder = tokenizer.apply_chat_template(messages_placeholder, add_generation_prompt=False)
#     placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)

#     # 3. Find the differing section.
#     # This works by finding the first point of difference between the two tokenized lists.
#     for i in range(min(len(ids_concept), len(ids_placeholder))):
#         if ids_concept[i] != ids_placeholder[i]:
#             # This is where the concept word starts.
#             start_index = i
#             # Find where the placeholder token is in the second list.
#             try:
#                 placeholder_end_index = ids_placeholder.index(placeholder_id, start_index)
#             except ValueError:
#                 return None # Placeholder not found, something is wrong
                
#             # The number of tokens in the concept word is the difference in length
#             # between the differing sections of the two lists.
#             len_concept = len(ids_concept) - len(ids_placeholder) + 1
#             return start_index + len_concept - 1

#     return None

# # --- Activation Extraction (Now with better progress reporting) ---
# def get_concept_token_activations(model, tokenizer, concept_prompts_data, layer_idx):
#     all_activations = {}
#     print(f"Extracting concept token activations from layer {layer_idx}...")
    
#     for concept, prompts in concept_prompts_data.items():
#         concept_activations = []
#         found_count = 0
#         for prompt_text in tqdm(prompts, desc=f"Processing '{concept}' prompts"):
#             target_index = find_concept_token_index(tokenizer, prompt_text, concept)
#             if target_index is None:
#                 continue

#             found_count += 1
#             def hook_fn(module, input, output, index=target_index):
#                 concept_token_activation = output[0][:, index, :].detach().cpu()
#                 hook_activations.append(concept_token_activation)

#             hook_activations = []
#             hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
#             messages = [{"role": "user", "content": prompt_text}]
#             inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False).to(DEVICE)
#             with torch.no_grad():
#                 model(input_ids=inputs)
#             hook_handle.remove()
#             if hook_activations:
#                 concept_activations.append(hook_activations[0])

#         if concept_activations:
#             all_activations[concept] = torch.cat(concept_activations, dim=0)
#             print(f"  -> For '{concept}', successfully found concept word in {found_count}/{len(prompts)} prompts.")
#         else:
#             print(f"  -> For '{concept}', FAILED to find concept word in any of the {len(prompts)} prompts.")

#     return all_activations

# # --- Manifold Analysis (with a safety check) ---
# def analyze_manifolds(all_activations_by_concept):
#     concept_analysis = {}
#     print("Centering representation space...")
    
#     # Filter out any empty results before calculating centroids
#     valid_activations = {k: v for k, v in all_activations_by_concept.items() if v is not None and len(v) > 0}
#     if not valid_activations:
#         print("Error: No valid activations were collected. Cannot perform analysis.")
#         return {}

#     centroids = {concept: acts.mean(dim=0) for concept, acts in valid_activations.items()}
    
#     # --- SAFETY CHECK ---
#     if not centroids:
#         print("ERROR: Centroids dictionary is empty. Cannot proceed.")
#         return {}
    
#     global_centroid = torch.stack(list(centroids.values())).mean(dim=0)
#     centered_activations = {concept: acts - global_centroid for concept, acts in valid_activations.items()}
    
#     print("Finding effective eigenvectors via PCA...")
#     for concept, acts in centered_activations.items():
#         if len(acts) < 2: continue
#         pca = PCA()
#         pca.fit(acts.numpy())
#         eigenvectors, eigenvalues = pca.components_, pca.explained_variance_
#         mean_eigval, std_eigval = eigenvalues.mean(), eigenvalues.std()
#         threshold = mean_eigval - 1.0 * std_eigval
#         effective_mask = eigenvalues > threshold
#         print(f"Concept '{concept}': Found {np.sum(effective_mask)} effective eigenvectors out of {min(len(acts)-1, len(eigenvalues))} (threshold: {threshold:.4f})")
#         concept_analysis[concept] = {
#             "pca": pca, "eigenvectors": torch.tensor(eigenvectors, dtype=torch.float32),
#             "eigenvalues": torch.tensor(eigenvalues, dtype=torch.float32), "effective_mask": torch.tensor(effective_mask, dtype=torch.bool),
#             "centered_acts": acts
#         }
#     return concept_analysis

# # --- Perturbation and Generation (Unchanged, but now receives correct indices) ---
# def generate_with_perturbation_on_concept(model, tokenizer, messages, layer_idx, direction, magnitude, eigenvalue, concept_word):
#     perturbation = direction * magnitude * torch.sqrt(eigenvalue)
#     user_prompt = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
#     if not user_prompt: return "Error: User prompt not found."
#     target_index = find_concept_token_index(tokenizer, user_prompt, concept_word)
    
#     prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)

#     if target_index is None:
#         print(f"\nWarning: Concept word '{concept_word}' not found in test prompt. Generating without perturbation.")
#         with torch.no_grad():
#             output_ids = model.generate(**inputs, max_new_tokens=70, do_sample=False, pad_token_id=tokenizer.eos_token_id)
#     else:
#         perturbation_applied = False
#         def hook_fn_modify(module, input, output, index=target_index):
#             nonlocal perturbation_applied
#             if not perturbation_applied:
#                 hidden_states = output[0]
#                 if index < hidden_states.shape[1]:
#                     hidden_states[:, index, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
#                 else:
#                     print(f"Warning: Index {index} out of bounds for shape {hidden_states.shape[1]} during hook. Skipping perturbation.")
#                 perturbation_applied = True
#                 return (hidden_states,) + output[1:]
#             return output
        
#         hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_modify)
#         with torch.no_grad():
#             output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
#         hook_handle.remove()

#     prompt_length = inputs['input_ids'].shape[1]
#     return tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)

# # --- Find Top Prompts (with safety check) ---
# def find_top_prompts(prompts, centered_acts, direction, n=10):
#     if centered_acts is None or len(centered_acts) == 0:
#         return {"positive": ["N/A - No activations found."], "negative": ["N/A - No activations found."]}
#     projections = centered_acts @ direction
#     _, sorted_indices = torch.sort(projections, descending=False)
#     # Ensure n is not larger than the number of available prompts
#     n = min(n, len(sorted_indices))
#     neg_indices, pos_indices = sorted_indices[:n], sorted_indices[-n:].flip(dims=[0])
#     return {"positive": [prompts[i] for i in pos_indices], "negative": [prompts[i] for i in neg_indices]}

# # --- Main Execution ---
# def main():
#     print("Loading prompt data from `prompts.json`...")
#     try:
#         with open('prompts.json', 'r', encoding='utf-8') as f:
#             concept_prompts_data = json.load(f)
#     except FileNotFoundError:
#         print("Error: `prompts.json` not found. Please ensure it's in the same directory.")
#         return

#     model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
#     all_activations = get_concept_token_activations(model, tokenizer, concept_prompts_data, TARGET_LAYER)
    
#     gc.collect(); torch.cuda.empty_cache()

#     analysis_results = analyze_manifolds(all_activations)
#     if not analysis_results:
#         print("Analysis failed because no valid activation data was collected. Exiting.")
#         return

#     test_concept = "dog"
#     user_prompt = f"A dog is"
#     system_prompt = "You are a language model assistant. Please continue this input:"

#     print("\n" + "="*80)
#     print(f"--- INTERVENTION ON CONCEPT TOKEN: '{test_concept}' ---")
#     print(f"--- LAYER: {TARGET_LAYER} ---")
#     print("="*80)

#     if test_concept not in analysis_results:
#         print(f"Analysis for concept '{test_concept}' not available. Exiting.")
#         return

#     concept_analysis = analysis_results[test_concept]
#     pc1_direction = concept_analysis["eigenvectors"][0]
#     pc1_eigenvalue = concept_analysis["eigenvalues"][0]
#     messages_to_test = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
#     print(f"\nSystem Prompt: '{system_prompt}'")
#     print(f"User Prompt:   '{user_prompt}'")
#     original_output = generate_with_perturbation_on_concept(model, tokenizer, messages_to_test, TARGET_LAYER, pc1_direction, 0, pc1_eigenvalue, test_concept)
#     print(f"Original model completion: ... {original_output}")
    
#     perturbation_scales = [-10.0, -5.0, -2.5, -1.5, 1.5, 2.5, 5.0, 10.0]
#     print(f"\n--- Perturbing '{test_concept}' token activation along PC1 ---")
#     for scale in perturbation_scales:
#         perturbed_output = generate_with_perturbation_on_concept(model, tokenizer, messages_to_test, TARGET_LAYER, pc1_direction, scale, pc1_eigenvalue, test_concept)
#         print(f"Perturbation scale {scale:+.1f}x: ... {perturbed_output}")

#     print("\n" + "="*80)
#     print(f"--- Analyzing original dataset prompts along the PC1 '{test_concept}' direction (Layer {TARGET_LAYER}) ---")
    
#     # This assumes the order of activations in 'centered_acts' matches the order in 'concept_prompts_data'.
#     # This holds true because we process prompts sequentially.
#     valid_prompts = concept_prompts_data[test_concept]
    
#     top_prompts = find_top_prompts(valid_prompts, concept_analysis["centered_acts"], pc1_direction, n=10)
#     print(f"\nTop 10 prompts most aligned with POSITIVE PC1 direction:")
#     for i, prompt in enumerate(top_prompts['positive'], 1): print(f"{i:2d}. '{prompt}'")
#     print(f"\nTop 10 prompts most aligned with NEGATIVE PC1 direction:")
#     for i, prompt in enumerate(top_prompts['negative'], 1): print(f"{i:2d}. '{prompt}'")
#     print("="*80)

# if __name__ == "__main__":
#     main()

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm
import gc
import json


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_LAYER = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

def get_model_and_tokenizer(model_name):
    """Loads the model and tokenizer."""
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PLACEHOLDER]']})
    model.resize_token_embeddings(len(tokenizer)) 
    print("Model and tokenizer loaded.")
    return model, tokenizer


def find_concept_token_index(tokenizer, prompt_text, concept_word):
    """
    Finds the index of the last token for a concept word using a robust placeholder method.
    """
    placeholder = "[PLACEHOLDER]"
    prompt_with_placeholder = prompt_text.replace(concept_word, placeholder, 1)

    messages_concept = [{"role": "user", "content": prompt_text}]
    messages_placeholder = [{"role": "user", "content": prompt_with_placeholder}]

    ids_concept = tokenizer.apply_chat_template(messages_concept, add_generation_prompt=False)
    ids_placeholder = tokenizer.apply_chat_template(messages_placeholder, add_generation_prompt=False)
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)

    for i in range(min(len(ids_concept), len(ids_placeholder))):
        if ids_concept[i] != ids_placeholder[i]:
            start_index = i
            try:
                placeholder_end_index = ids_placeholder.index(placeholder_id, start_index)
            except (ValueError, IndexError): return None
            len_concept = len(ids_concept) - len(ids_placeholder) + 1
            return start_index + len_concept - 1
    return None


def get_concept_token_activations(model, tokenizer, concept_prompts_data, layer_idx):
    """
    Extracts activations for the specific concept token in each prompt.
    BUG FIX: Now returns a dictionary of valid prompts alongside their activations.
    """
    all_results = {}
    print(f"Extracting concept token activations from layer {layer_idx}...")
    
    for concept, prompts in concept_prompts_data.items():
        concept_activations = []
        valid_prompts_for_concept = [] # Track prompts where the concept was found

        for prompt_text in tqdm(prompts, desc=f"Processing '{concept}' prompts"):
            target_index = find_concept_token_index(tokenizer, prompt_text, concept)
            if target_index is None:
                continue

            valid_prompts_for_concept.append(prompt_text) # This prompt is valid

            def hook_fn(module, input, output, index=target_index):
                hook_activations.append(output[0][:, index, :].detach().cpu())

            hook_activations = []
            hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
            messages = [{"role": "user", "content": prompt_text}]
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=False).to(DEVICE)
            with torch.no_grad():
                model(input_ids=inputs)
            hook_handle.remove()
            if hook_activations:
                concept_activations.append(hook_activations[0])

        if concept_activations:
            all_results[concept] = {
                "activations": torch.cat(concept_activations, dim=0),
                "prompts": valid_prompts_for_concept
            }
            print(f"  -> For '{concept}', successfully found concept in {len(valid_prompts_for_concept)}/{len(prompts)} prompts.")
        else:
            print(f"  -> For '{concept}', FAILED to find concept in any of the {len(prompts)} prompts.")

    return all_results


def analyze_manifolds(all_results):
    """
    Performs PCA on the collected activations.
    The note about centering: Subtracting the global mean helps isolate directions
    unique to a concept (e.g., "dogness") from directions common to all concepts
    (e.g., "being a noun" or "being an animal"). It's a standard technique to
    improve the specificity of the principal components.
    """
    concept_analysis = {}
    
    valid_activations = {k: v['activations'] for k, v in all_results.items() if v and len(v['activations']) > 0}
    if not valid_activations:
        print("Error: No valid activations collected. Cannot perform analysis.")
        return {}

    print("Centering representation space...")
    centroids = {concept: acts.mean(dim=0) for concept, acts in valid_activations.items()}
    global_centroid = torch.stack(list(centroids.values())).mean(dim=0)
    centered_activations = {concept: acts - global_centroid for concept, acts in valid_activations.items()}
    
    for concept, acts in centered_activations.items():
        if len(acts) < 2: continue
        pca = PCA()
        pca.fit(acts.numpy())
        eigenvectors, eigenvalues = pca.components_, pca.explained_variance_
        concept_analysis[concept] = {
            "pca": pca,
            "eigenvectors": torch.tensor(eigenvectors, dtype=torch.float32),
            "eigenvalues": torch.tensor(eigenvalues, dtype=torch.float32),
            "centered_acts": acts,
            "valid_prompts": all_results[concept]['prompts']
        }
    return concept_analysis


def generate_with_perturbation_on_concept(model, tokenizer, messages, layer_idx, direction, magnitude, eigenvalue, concept_word):
    """Generates text while perturbing the concept token's activation."""
    perturbation = direction * magnitude * torch.sqrt(eigenvalue)
    user_prompt = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
    if not user_prompt: return "Error: User prompt not found."
    target_index = find_concept_token_index(tokenizer, user_prompt, concept_word)
    
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)

    if target_index is None:
        print(f"\nWarning: Concept word '{concept_word}' not found in test prompt. Generating without perturbation.")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    else:
        perturbation_applied = False
        def hook_fn_modify(module, input, output, index=target_index):
            nonlocal perturbation_applied
            if not perturbation_applied:
                hidden_states = output[0]
                if index < hidden_states.shape[1]:
                    hidden_states[:, index, :] += perturbation.to(hidden_states.device, dtype=hidden_states.dtype)
                perturbation_applied = True
                return (hidden_states,) + output[1:]
            return output
        
        hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn_modify)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        hook_handle.remove()

    prompt_length = inputs['input_ids'].shape[1]
    return tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)


def find_top_prompts(valid_prompts, centered_acts, direction, n=10):
    """
    Finds top N prompts.
    BUG FIX: Takes `valid_prompts` as input to ensure indices match.
    """
    if centered_acts is None or len(centered_acts) < n:
        return {"positive": ["N/A"], "negative": ["N/A"]}
    projections = centered_acts @ direction
    _, sorted_indices = torch.sort(projections, descending=False)
    neg_indices, pos_indices = sorted_indices[:n], sorted_indices[-n:].flip(dims=[0])
    return {
        "positive": [valid_prompts[i] for i in pos_indices],
        "negative": [valid_prompts[i] for i in neg_indices]
    }


def main():
    print("Loading prompt data from `prompts.json`...")
    try:
        with open('prompts.json', 'r', encoding='utf-8') as f:
            concept_prompts_data = json.load(f)
    except FileNotFoundError:
        print("Error: `prompts.json` not found. Please ensure it's in the same directory.")
        return

    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    all_results = get_concept_token_activations(model, tokenizer, concept_prompts_data, TARGET_LAYER)
    
    gc.collect(); torch.cuda.empty_cache()
    analysis_results = analyze_manifolds(all_results)
    if not analysis_results:
        print("Analysis failed because no valid data was collected. Exiting.")
        return

    test_concept = "dog"
    user_prompt = f"I went to the park and saw a large {test_concept} chasing a frisbee."
    system_prompt = "You are a helpful assistant. Continue the story directly and concisely."

    print("\n" + "="*80)
    print(f"--- INTERVENTION ON CONCEPT TOKEN: '{test_concept}' ---")
    print(f"--- LAYER: {TARGET_LAYER} ---")
    print("="*80)

    if test_concept not in analysis_results:
        print(f"Analysis for concept '{test_concept}' not available. Exiting.")
        return

    concept_analysis = analysis_results[test_concept]
    pc1_direction = concept_analysis["eigenvectors"][0]
    pc1_eigenvalue = concept_analysis["eigenvalues"][0]
    messages_to_test = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    original_output = generate_with_perturbation_on_concept(model, tokenizer, messages_to_test, TARGET_LAYER, pc1_direction, 0, pc1_eigenvalue, test_concept)
    print(f"Original model completion: ... {original_output}")
    
    perturbation_scales = [-5.0, -2.5, -1.5, 1.5, 2.5, 5.0]
    print(f"\n--- Perturbing '{test_concept}' token activation along PC1 ---")
    for scale in perturbation_scales:
        perturbed_output = generate_with_perturbation_on_concept(model, tokenizer, messages_to_test, TARGET_LAYER, pc1_direction, scale, pc1_eigenvalue, test_concept)
        print(f"Perturbation scale {scale:+.1f}x: ... {perturbed_output}")

    print("\n" + "="*80)
    print(f"--- Analyzing original dataset prompts along the PC1 '{test_concept}' direction (Layer {TARGET_LAYER}) ---")
    

    top_prompts = find_top_prompts(
        concept_analysis['valid_prompts'],
        concept_analysis['centered_acts'],
        pc1_direction,
        n=10
    )
    
    print(f"\nTop 10 prompts most aligned with POSITIVE PC1 direction:")
    for i, prompt in enumerate(top_prompts['positive'], 1): print(f"{i:2d}. '{prompt}'")
    print(f"\nTop 10 prompts most aligned with NEGATIVE PC1 direction:")
    for i, prompt in enumerate(top_prompts['negative'], 1): print(f"{i:2d}. '{prompt}'")
    print("="*80)

if __name__ == "__main__":
    main()