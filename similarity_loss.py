import torch

#Slightly modified from 
#Kovaleva et al. (2018) 'Similarity-Based Reconstruction Loss for Meaning Representation'
#paper link: https://www.aclweb.org/anthology/D18-1525/
#code link: https://github.com/text-machine-lab/similarity-loss/blob/master/loss.py
#changes: log_softmax for stability, removed padding bit, added device, added similarities option
#so you can load
#TODO: will have to adjust vocab so that I don't have to do unk thing in match embeddings util
#TODO: refactor so i can do this dynamically so as to learn space at same time
class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    The per-example loss takes predicted probabilities for every word in a vocabulary and penalizes the model
    for predicting high probabilities for words that are semantically distant from the target word.
        Args:
        embeddings (torch.FloatTensor): pre-trained word embeddings
        device: device to use
    """
    def __init__(self, embeddings, device, similarities=None):
        super().__init__()

        #not using presaved
        if similarities is None:
            voc_size = embeddings.shape[0]

            # Compute similarities
            print("Computing word similarities...")
            similarities = []
            for i in range(voc_size):
                similarities.append(torch.nn.functional.cosine_similarity(embeddings[i].expand_as(embeddings), embeddings))

        similarities = torch.stack(similarities).to(device)
        self.similarities = similarities

    def forward(self, output, targets):
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.mean(torch.sum(-log_probs*self.similarities[targets], 1))
        return loss

