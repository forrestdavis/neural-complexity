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
    def __init__(self, similarities=None, neg_sim=False, 
            normalize_sim=True, N_sim=2, stop_idx=torch.tensor([])):
        super().__init__()

        if similarities is None:
            self.similarities = None
        else:
            self.similarities = similarities
        self.stop_idx = stop_idx
        self.N_sim = N_sim
        self.neg_sim = neg_sim
        self.normalize_sim=normalize_sim

    def forward(self, output, targets, embeddings=None):

        if self.similarities is None:
            with torch.no_grad():
                #Get cosine similarities using half precision to save GPU memory
                NORM = torch.norm(embeddings, dim=1).half()
                denom = torch.einsum('i,j', NORM, NORM).half()
                similarities = torch.einsum('ij,kj -> ik', embeddings, embeddings).half()
                similarities.div_(NORM)

                #Free up the memory
                del(NORM)
                del(denom)

                #Clip
                if not self.neg_sim:
                    torch.nn.functional.relu(similarities, inplace=True)

                #If ignoring stop words
                if self.stop_idx.shape[0] != 0:
                    similarities[self.stop_idx,:] = 0
                    similarities[self.stop_idx,self.stop_idx]=1

                #TOP N
                mask = torch.zeros(similarities.shape, dtype=torch.uint8).to(embeddings.device)
                mask.scatter_(1, torch.topk(similarities, self.N_sim, dim=1).indices.to(embeddings.device), 1)
                similarities.mul_(mask)

                del(mask)

                #Normalize by row
                if self.normalize_sim:
                    similarities.div_(torch.sum(similarities, dim=1, keepdim=True))
        else:
            similarities = self.similarities

        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        loss = torch.mean(torch.sum(-log_probs*similarities[targets], 1))
        return loss



    def get_cohort(self, embeddings, token_id=287):

        with torch.no_grad():
            #Get cosine similarities
            NORM = torch.norm(embeddings, dim=1)
            denom = torch.einsum('i,j', NORM, NORM)
            similarities = torch.einsum('ij,kj -> ik', embeddings, embeddings)
            similarities.div_(NORM)

            #Free up the memory
            del(NORM)
            del(denom)

            #Clip
            if not self.neg_sim:
                torch.nn.functional.relu(similarities, inplace=True)

            #If ignoring stop words
            if self.stop_idx.shape[0] != 0:
                similarities[self.stop_idx,:] = 0
                similarities[self.stop_idx,self.stop_idx]=1

            #TOP N
            mask = torch.zeros(similarities.shape, dtype=torch.uint8).to(embeddings.device)
            mask.scatter_(1, torch.topk(similarities, self.N_sim, dim=1).indices.to(embeddings.device), 1)
            similarities.mul_(mask)

            del(mask)

            #Normalize by row
            if self.normalize_sim:
                similarities.div_(torch.sum(similarities, dim=1, keepdim=True))

        top = torch.topk(similarities, 10, dim=1)

        return top.values[token_id], top.indices[token_id]
