import numpy as np
import math

class Pop:
    def __init__(self, gene_sequence):
        self.gene_sequence = gene_sequence
        self.score = 0.0
        self.model_data = None
    
class Generation:
    def __init__(self, genes, fit_fn, reproduction_strat):
        self.pops = [Pop(gene) for gene in genes]
        self.n_pops = len(self.pops)
        self.fit_fn = fit_fn
        self.reproduction_strat = reproduction_strat

    def best_score(self): return self.pops[0].score

    def fit(self, data):
        for pid in range(self.n_pops):
            self.pops[pid] = self.fit_fn(data, self.pops[pid])
        self.pops = sorted(self.pops, key=lambda pop: pop.score, reverse=True)

    def reproduce(self, target_pop_size=-1, mutation_rate=0.0):
        """
        mutation_rate: randomly flip 1 bit of the binary encoded gene to a random selection of the population specified by mutation_rate
        """
        n_elites = self.n_pops % 2
        survived_genes = [self.pops[i].gene_sequence for i in range(n_elites)]
        remain_indexes = np.arange(n_elites, self.n_pops)
        for i in range(self.n_pops // 2):
            match_filter = np.sum(
                np.eye(len(remain_indexes))[
                    np.random.choice(len(remain_indexes), 2, replace=False)
                ], axis=0
            )
            picked_for_match = remain_indexes[match_filter == 1]
            remain_indexes = remain_indexes[match_filter == 0]
            survived_genes.append(
                self.pops[picked_for_match[0]].gene_sequence
                    if self.pops[picked_for_match[0]].score > self.pops[picked_for_match[1]].score
                    else self.pops[picked_for_match[1]].gene_sequence
            )

        n_survived_genes = len(survived_genes)
        survived_genes = np.array(survived_genes)

        if (target_pop_size < 0): target_pop_size = self.n_pops // 2

        pair_maps = []
        next_genes = []

        for i in range(int(np.min([
            math.comb(n_survived_genes, 2), target_pop_size // 2
        ]))):
            pair_map = np.sum(
                np.eye(n_survived_genes)[
                    np.random.choice(n_survived_genes, 2, replace=False)
                ], axis=0
            )
            while has_duplicate(pair_map, pair_maps):
                pair_map = np.sum(
                    np.eye(n_survived_genes)[
                        np.random.choice(n_survived_genes, 2, replace=False)
                    ], axis=0
                )

            pair = survived_genes[pair_map == 1]

            offsprings = self.reproduction_strat(pair[0], pair[1])

            for offspring in offsprings:
                if has_duplicate(offspring, next_genes): continue
                next_genes.append(offspring)

            pair_maps.append(pair_map)

        n_mutations = round(len(next_genes) * mutation_rate)
        mutation_map = np.random.choice(len(next_genes), n_mutations, replace=False)
        gene_seq_len = 0 if n_mutations == 0 else len(next_genes[0])
        mutation_filters = None \
            if n_mutations == 0 \
            else np.eye(gene_seq_len)[np.random.choice(gene_seq_len, n_mutations, replace=True)]

        for i, mid in enumerate(mutation_map):
            next_genes[mid] = next_genes[mid] ^ mutation_filters[i].astype(int)
        
        return Generation(next_genes, self.fit_fn, self.reproduction_strat)

class GA:
    def __init__(self,
        adam_n_eve, # the root gene pair that will be used to produce generation 0
        data, 
        fit_fn, 
        init_pop_size, 
        reproduction_strat, 
        max_n_generations=10
        # random_seed=0
    ):
        # self.rng = np.random.rng(random_seed)
        adam, eve = adam_n_eve
        i_length = len(adam)

        init_gene_filters = []

        for i in range(int(np.min([
            math.comb(i_length, (i_length // 2) + 1) * 0.9, 
            np.ceil(init_pop_size/2)
        ]))):
            _gene_filter = np.sum(
                np.eye(i_length)[
                    np.random.choice(i_length, (i_length // 2) + 1, replace=False)
                ], axis=0
            )
            while has_duplicate(_gene_filter, init_gene_filters):
                _gene_filter = np.sum(
                    np.eye(i_length)[
                        np.random.choice(i_length, (i_length // 2) + 1, replace=False)
                    ], axis=0
                )

            init_gene_filters.append(_gene_filter)

        init_genes = np.concatenate(
            [gene_blend(adam, eve, f) for f in init_gene_filters], axis=0
        )[:init_pop_size]

        self.generations = [Generation(init_genes, fit_fn, reproduction_strat)]
        self.data = data
        self.max_n_generations = max_n_generations
        self.n_generations = 1

    def start(self, mutation_rate):
        for gen in range(self.max_n_generations):
            self.generations[-1].fit(self.data)
            if (self.n_generations > 1):
                if (self.generations[-2].best_score() > self.generations[-1].best_score()):
                    return
                if (self.generations[-1].n_pops < 4):
                    return

            self.generations.append(self.generations[-1].reproduce(-1, mutation_rate))
            self.n_generations += 1

def has_duplicate(gene, gene_pool):
    for f in gene_pool:
        # if (~(np.logical_xor(gene, f)).any() or 
        #     (np.logical_xor(gene, f)).all()):
        #     return True
        if (~(np.logical_xor(gene, f)).any()): return True
    return False

def gene_blend(gene_1, gene_2, blend_filter):
    return [
        np.logical_or(gene_1 * blend_filter, gene_2 * np.logical_not(blend_filter)).astype(int), 
        np.logical_or(gene_2 * blend_filter, gene_1 * np.logical_not(blend_filter)).astype(int)
    ]