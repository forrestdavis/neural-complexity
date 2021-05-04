import torch

def chunked_pairwise_cosine_similarity(x1, x2=None, eps=1e-8, MAXIMUM_SIZE=33000):
    x1.half()

    if x1.shape[0] > MAXIMUM_SIZE:
        return_tensor = torch.zeros((x1.shape[0], x1.shape[0])).half().to(x1.device)

        #chunk_size = torch.remainder(x1.shape, torch.tensor(MAXIMUM_SIZE))
        x1_chunks = list(torch.split(x1, MAXIMUM_SIZE))
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True).half()
        w1_chunks = list(torch.split(w1, MAXIMUM_SIZE))
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True).half()

        del(x1)
        del(w1)

        start_idx = 0
        while x1_chunks:
            x1_chunk = x1_chunks.pop(0)
            w1_chunk = w1_chunks.pop(0)
            end_idx = start_idx + x1_chunk.shape[0]
            return_tensor[start_idx:end_idx] = torch.mm(x1_chunk, x2.t())/(w1_chunk*w2.t()).clamp(min=eps)
            start_idx = end_idx
            del(x1_chunk)
            del(w1_chunk)
        return return_tensor

    else:
        x1.half()
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True).half()
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True).half()
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

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

                #Get cosine similarities
                similarities = chunked_pairwise_cosine_similarity(embeddings.clone().detach())

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
            similarities = chunked_pairwise_cosine_similarity(embeddings.clone().detach())

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
            #if self.normalize_sim:
            #    similarities.div_(torch.sum(similarities, dim=1, keepdim=True))

        top = torch.topk(similarities, self.N_sim, dim=1)

        return top.values[token_id], top.indices[token_id]
