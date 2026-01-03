# organizer_scripts/build_dataset.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Load expression data
# -----------------------------
expr_cfRNA = pd.read_csv('data/expr_df_2_GSE192902.csv', index_col=0)
expr_placenta = pd.read_csv('data/expr_df_GSE234729.csv', index_col=0)

meta_cfRNA = pd.read_csv('data/metadata_cfRNA.csv', index_col=0)
meta_placenta = pd.read_csv('data/metadata_placenta.csv', index_col=0)

# -----------------------------
# 2. Gene harmonization
# -----------------------------
shared_genes = expr_cfRNA.columns.intersection(expr_placenta.columns)
expr_cfRNA = expr_cfRNA[shared_genes]
expr_placenta = expr_placenta[shared_genes]

# -----------------------------
# 3. Normalize expression per gene
# -----------------------------
scaler = StandardScaler()
expr_cfRNA_scaled = pd.DataFrame(scaler.fit_transform(expr_cfRNA), 
                                 index=expr_cfRNA.index, columns=shared_genes)
expr_placenta_scaled = pd.DataFrame(scaler.transform(expr_placenta), 
                                    index=expr_placenta.index, columns=shared_genes)

# -----------------------------
# 4. Create train.csv (cfRNA) and test.csv (placenta)
# -----------------------------
train_df = expr_cfRNA_scaled.copy()
train_df['target'] = meta_cfRNA['diagnosis']  # 1=preeclampsia, 0=control
train_df['node_id'] = ['cfRNA_'+str(i) for i in range(len(train_df))]
train_df = train_df.reset_index(drop=True)
train_df.to_csv('data/train.csv', index=False)

test_df = expr_placenta_scaled.copy()
test_df['node_id'] = ['placenta_'+str(i) for i in range(len(test_df))]
test_df = test_df.reset_index(drop=True)
test_df.to_csv('data/test.csv', index=False)

# -----------------------------
# 5. Node types
# -----------------------------
node_types = pd.DataFrame({
    'node_id': list(train_df['node_id']) + list(test_df['node_id']),
    'node_type': ['cfRNA']*len(train_df) + ['placenta']*len(test_df)
})
node_types.to_csv('data/node_types.csv', index=False)

# -----------------------------
# 6. Build graph_edges.csv
# -----------------------------
edges = []

# a) Within-modality cosine similarity edges (sparse)
def build_edges(df, node_prefix, top_k=10):
    sim_matrix = cosine_similarity(df[shared_genes])
    for i in range(sim_matrix.shape[0]):
        top_idx = np.argsort(sim_matrix[i])[-(top_k+1):-1]  # skip self
        for j in top_idx:
            edges.append({
                'src': f'{node_prefix}_{i}',
                'dst': f'{node_prefix}_{j}',
                'edge_type': 'similarity'
            })

build_edges(train_df, 'cfRNA')
build_edges(test_df, 'placenta')

# b) Cross-modality edges (cfRNA <-> placenta)
sim_matrix_cross = cosine_similarity(train_df[shared_genes], test_df[shared_genes])
top_k = 5
for i in range(sim_matrix_cross.shape[0]):
    top_idx = np.argsort(sim_matrix_cross[i])[-top_k:]
    for j in top_idx:
        edges.append({
            'src': f'cfRNA_{i}',
            'dst': f'placenta_{j}',
            'edge_type': 'cross_modality'
        })

# c) Optional: ancestry / gestational age edges
# Example for ancestry: connect nodes of same ancestry within each dataset
for df, prefix in zip([meta_cfRNA, meta_placenta], ['cfRNA','placenta']):
    ancestry_map = df['ancestry'].to_dict()
    nodes = list(df.index)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if ancestry_map[nodes[i]] == ancestry_map[nodes[j]]:
                edges.append({
                    'src': f'{prefix}_{i}',
                    'dst': f'{prefix}_{j}',
                    'edge_type': 'ancestry'
                })

# Convert to DataFrame and save
edges_df = pd.DataFrame(edges)
edges_df.to_csv('data/graph_edges.csv', index=False)

print("✅ Dataset built successfully: train.csv, test.csv, node_types.csv, graph_edges.csv")
