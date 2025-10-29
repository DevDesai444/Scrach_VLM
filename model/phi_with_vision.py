import torch
import torch.nn as nn

class PhiWithVision(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.phi = base_model
        
        # Add a projection layer for image embeddings
        self.image_projection = nn.Linear(
            in_features=512,  # SigLIP embedding dimension
            out_features=self.phi.config.hidden_size  # Phi model dimension
        )
        
        # Copy necessary attributes from base model
        self.config = base_model.config
        self.prepare_inputs_for_generation = base_model.prepare_inputs_for_generation
        self.generation_config = base_model.generation_config
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, input_ids=None, attention_mask=None, image_embeddings=None, 
                labels=None, inputs_embeds=None, **kwargs):
        # If inputs_embeds is provided, use it directly
        if inputs_embeds is not None:
            combined_embeddings = inputs_embeds
        else:
            # Project image embeddings to match model dimensions
            projected_image = self.image_projection(image_embeddings)
            
            # Get text embeddings from first layer
            text_embeddings = self.phi.get_input_embeddings()(input_ids)
            
            # Concatenate embeddings
            combined_embeddings = torch.cat([projected_image, text_embeddings], dim=1)
            
            # Adjust attention mask
            if attention_mask is not None:
                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device
                )
                attention_mask = torch.cat([extended_attention_mask, attention_mask], dim=1)
            
            # Adjust labels to match input length
            if labels is not None:
                # Truncate labels to match input length
                labels = labels[:, :input_ids.shape[1]]  # Truncate to input length
                
                # Create padding label for the image token position
                image_token_labels = torch.full(
                    (labels.shape[0], 1),
                    -100,  # Ignore index for loss calculation
                    device=labels.device,
                    dtype=labels.dtype
                )
                labels = torch.cat([image_token_labels, labels], dim=1)
        
        # Forward pass through the model
        outputs = self.phi(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs

    def get_input_embeddings(self):
        return self.phi.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.phi.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.phi.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.phi.set_output_embeddings(new_embeddings) 