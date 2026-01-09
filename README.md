
# 🧬 GNN Challenge: cfRNA → Placenta Inductive GNN for Maternal-Fetal Health Prediction

<div align="center">
    <img src="images/IMG1.png" width='650' /> 
</div>

<br>
<br


### Scientific Focus
- Inductive graph learning across **cfRNA** and **placental transcriptomics** to detect maternal-fetal health issues.  
- Learn **transferable representations** that generalize to unseen samples and domains rather than treating each dataset independently.  

### Alignment with [BASIRA's](https://basira-lab.com) Mission
- Prioritizes robust generalization across heterogeneous datasets.  
- Uses compute-efficient, non–data-hungry graph learning methods** that can run on standard hardware.  

### Inspiration from GNN Literature
- Draws from studies on inductive learning, message passing, and representation transfer**.  
- Model design follows [**DGL Lectures 1.1–4.6**](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T), covering:  
  - Graph construction from tabular data  
  - Node feature encoding  
  - Neighborhood aggregation (**GraphSAGE-style inductive updates**)  
  - Mini-batch training via **neighborhood sampling**  
  - Inductive inference on unseen nodes  

---


## Dataset Source and Description

### Source
- Publicly available on **Gene Expression Omnibus (GEO)**, maintained by the **NIH**.   

### Datasets Used
- **Maternal plasma cfRNA data:** [`GSE192902`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE192902)
- **Placental RNA-seq data:** [`GSE234729`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7478653)  
- **Features:** 6,000 harmonized gene expression features across two cell types  

### Training and Test Data
- **Training Data:** 209–210 cfRNA samples (balanced)  
- **Test Data:** 123–124 placenta samples (**inductive, unseen during training**)  
- **Classes:** 0 = Control, 1 = Preeclampsia  

### Purpose and Integration Goal
- Identify and validate cfRNA biomarkers for early prediction of preeclampsia, often before clinical symptoms appear.  
- Support research in maternal-fetal health and early detection of preeclampsia.  
- Integrate gene expression and clinical metadata to capture subtle risk patterns while handling noisy and imbalanced data for robust and equitable predictions.  

---

## Dataset Construction and Preprocessing [(`build_dataset.ipynb`)](./organizer_scripts/build_dataset.ipynb) and [Kaggle](https://www.kaggle.com/code/freeeman/maternal-2014425c3f4)


**Objective:** Ensure structural compatibility for graph construction and inductive learning by Hnadling Expression Data, Parsing and Cleaning Metedata, and Expression-Metadata Fusion  

**Steps Implemented:**  
- **Expression Data Handling:** Load and align expression matrices (sample × gene).  
- **Metadata Parsing and Cleaning:** Normalize clinical and biological attributes; clean string-based metadata.  
- **Expression–Metadata Fusion:** Merge expression and metadata tables using sample IDs to form node-level feature matrices.  

**Dataset Properties and Complexity:**  
- Small enough for local download yet challenging: high-dimensional features, rich but noisy metadata, biological heterogeneity.  

**Constraints:**  
- No external data  
- Fixed feature space  
- Inductive setting: test samples unseen during training  
- Feasible on standard hardware  

**Deliberate Complexity:**  
- Noise and missingness in metadata  
- Unbalanced disease labels  
- Predictive patterns emerge only through **neighborhood aggregation**  
- Large feature space vs. sample size requiring inductive bias  
- Cross-dataset domain shift (cfRNA vs. placenta) requiring **generalizable representations**  

---


## Advanced GNN Implementation [(`advanced_GNN_model.py`)](./starter_code/advanced_GNN_model.py)


<div align="center">
    <img src="images/IMG2.png" width='650' /> 
</div>

<br>
<br

**Objective:** Implement an advanced inductive GNN for cfRNA → placenta prediction, ensuring generalizable node representations and inductive learning.  


**Key Components:**  
- **Graph Construction:** Build hetero-graphs using similarity and ancestry edges.  
- **Node Feature Encoding:** Integrate gene expression and metadata into node-level features.  
- **Neighborhood Aggregation:** GraphSAGE-style layers with BatchNorm and ReLU for neighbor information propagation.  
- **Mini-Batch Training:** Use neighborhood sampling** for efficient training on large graphs.  
- **Inductive Inference:** Generate predictions for unseen placenta nodes without label leakage.  



## 📝 Citation

If you use this challenge or dataset in your research, please cite:

```bibtex
@dataset{gnn_challenge_2026,
  title={GNN Challenge: cfRNA → Placenta Inductive GNN for Maternal-Fetal Health Prediction},
  author={Mubaraq Onipede},
  year={2026},
  url={https://github.com/Mubarraqqq/gnn-challenge}
}
```

---

## 📄 License

See [LICENSE](LICENSE) file for details.



**Challenge Status**: ✅ Active  
**Leaderboard**: Live & Auto-updating  
**Submissions**: Open via GitHub PRs  
**Last Updated**: January 7, 2026

**Good luck! 🚀 We look forward to your submissions!**

