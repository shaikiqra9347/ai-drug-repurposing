
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# =========================
# CONFIGURATION
# =========================

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

st.set_page_config(
    page_title="AI Drug Repurposing",
    page_icon="💊",
    layout="wide"
)

# =========================
# OPEN TARGETS FUNCTIONS
# =========================

def find_drug(drug_name):

    query = """
    query SearchDrug($q: String!) {
      search(queryString: $q, entityNames: ["drug"]) {
        hits {
          id
          name
          entity
        }
      }
    }
    """

    response = requests.post(
        OT_URL,
        json={
            "query": query,
            "variables": {"q": drug_name}
        },
        timeout=30
    )

    data = response.json()

    hits = data.get("data", {}).get("search", {}).get("hits", [])

    if not hits:
        return None

    return hits[0]


def get_drug_targets(drug_id):

    query = """
    query DrugMechanisms($id: String!) {
      drug(chemblId: $id) {
        id
        name
        mechanismsOfAction {
          rows {
            mechanismOfAction
            targets {
              id
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
    """

    response = requests.post(
        OT_URL,
        json={
            "query": query,
            "variables": {"id": drug_id}
        },
        timeout=30
    )

    data = response.json()

    drug = data.get("data", {}).get("drug")

    if not drug:
        return []

    targets = []

    for row in drug.get("mechanismsOfAction", {}).get("rows", []):

        mechanism = row.get("mechanismOfAction", "")

        for target in row.get("targets", []):

            targets.append({
                "Target_ID": target.get("id"),
                "Target": target.get("approvedSymbol"),
                "Target_Name": target.get("approvedName"),
                "Mechanism": mechanism
            })

    return targets


def get_target_diseases(ensembl_id):

    query = """
    query TargetDiseases($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        associatedDiseases(page:{index:0,size:50}) {
          rows {
            score
            disease {
              id
              name
            }
          }
        }
      }
    }
    """

    response = requests.post(
        OT_URL,
        json={
            "query": query,
            "variables": {"ensemblId": ensembl_id}
        },
        timeout=30
    )

    data = response.json()

    target = data.get("data", {}).get("target")

    if not target:
        return []

    results = []

    for row in target.get("associatedDiseases", {}).get("rows", []):

        disease = row.get("disease")

        if disease:
            results.append({
                "disease_id": disease.get("id"),
                "disease_name": disease.get("name"),
                "score": row.get("score", 0)
            })

    return results


# =========================
# PUBMED
# =========================

def pubmed_search(drug, disease):

    term = f'"{drug}" AND "{disease}"'

    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": 5,
        "tool": "AI_Drug_Repurposing_App",
        "email": "example@example.com"
    }

    response = requests.get(
        PUBMED_URL,
        params=params,
        timeout=30
    )

    data = response.json()

    result = data.get("esearchresult", {})

    return {
        "count": int(result.get("count", 0)),
        "pmids": result.get("idlist", [])
    }


# =========================
# AI ANALYSIS
# =========================

def analyze_drug(drug_name):

    drug = find_drug(drug_name)

    if not drug:
        return None

    drug_id = drug.get("id")

    targets = get_drug_targets(drug_id)

    all_diseases = []

    for target in targets:

        target_id = target.get("Target_ID")

        if not target_id:
            continue

        diseases = get_target_diseases(target_id)

        for disease in diseases:

            disease["target_symbol"] = target.get("Target")
            disease["target_name"] = target.get("Target_Name")

            all_diseases.append(disease)

    if not all_diseases:
        return {
            "drug": drug,
            "targets": targets,
            "ranking": pd.DataFrame()
        }

    df = pd.DataFrame(all_diseases)

    ranking = (
        df.groupby(["disease_id", "disease_name"])
        .agg(
            Mean_Association_Score=("score", "mean"),
            Max_Association_Score=("score", "max"),
            Supporting_Targets=("target_symbol", "nunique")
        )
        .reset_index()
    )

    ranking["AI_Repurposing_Score"] = (
        0.60 * ranking["Max_Association_Score"]
        + 0.30 * ranking["Mean_Association_Score"]
        + 0.10 * (
            ranking["Supporting_Targets"] / 5
        ).clip(upper=1)
    )

    ranking = ranking.sort_values(
        "AI_Repurposing_Score",
        ascending=False
    ).reset_index(drop=True)

    ranking["Rank"] = ranking.index + 1

    return {
        "drug": drug,
        "targets": targets,
        "diseases": df,
        "ranking": ranking
    }


# =========================
# APP HEADER
# =========================

st.title("💊 AI-Driven Drug Repurposing Dashboard")

st.markdown(
    """
    ### From Drug → Molecular Targets → Disease Associations → AI Ranking → Literature Validation

    **Purpose:** computational hypothesis generation for potential drug repurposing.
    """
)

st.divider()


# =========================
# SIDEBAR
# =========================

st.sidebar.header("🔬 Drug Analysis")

drug_input = st.sidebar.text_input(
    "Enter a drug",
    value="Aspirin"
)

analyze_button = st.sidebar.button(
    "🚀 Analyze Drug",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Example drugs: Aspirin, Metformin, Ibuprofen, "
    "Paracetamol, Atorvastatin, Dexamethasone"
)


# =========================
# ANALYSIS
# =========================

if analyze_button:

    with st.spinner("Running AI drug repurposing analysis..."):

        try:

            result = analyze_drug(drug_input)

        except Exception as e:

            st.error(f"Analysis error: {e}")
            st.stop()

    if not result:

        st.error("Drug not found. Try another drug name.")

    else:

        drug = result["drug"]
        targets = result["targets"]
        ranking = result["ranking"]

        # =====================
        # DRUG INFORMATION
        # =====================

        st.header("1️⃣ Drug Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Drug",
                drug.get("name", drug_input)
            )

        with col2:
            st.metric(
                "ChEMBL ID",
                drug.get("id", "N/A")
            )

        # =====================
        # TARGETS
        # =====================

        st.header("2️⃣ Molecular Targets")

        if targets:

            targets_df = pd.DataFrame(targets)

            st.dataframe(
                targets_df,
                use_container_width=True,
                hide_index=True
            )

            st.metric(
                "Number of Targets",
                len(targets_df)
            )

        else:

            st.warning("No molecular targets found.")

        # =====================
        # AI RANKING
        # =====================

        if not ranking.empty:

            st.header("3️⃣ AI Repurposing Ranking")

            top10 = ranking.head(10).copy()

            display_df = top10[
                [
                    "Rank",
                    "disease_name",
                    "AI_Repurposing_Score",
                    "Supporting_Targets",
                    "Max_Association_Score"
                ]
            ].copy()

            display_df.columns = [
                "Rank",
                "Disease",
                "AI Repurposing Score",
                "Supporting Targets",
                "Max Association Score"
            ]

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # =====================
            # SCORE GRAPH
            # =====================

            st.subheader("📊 AI Repurposing Score")

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.barh(
                top10["disease_name"].iloc[::-1],
                top10["AI_Repurposing_Score"].iloc[::-1]
            )

            ax.set_xlabel("AI Repurposing Score")
            ax.set_ylabel("Disease")
            ax.set_title("Top Potential Repurposing Candidates")

            plt.tight_layout()

            st.pyplot(fig)

            # =====================
            # SUPPORTING TARGET GRAPH
            # =====================

            st.subheader("🎯 Supporting Molecular Targets")

            fig2, ax2 = plt.subplots(figsize=(10, 5))

            ax2.barh(
                top10["disease_name"].iloc[::-1],
                top10["Supporting_Targets"].iloc[::-1]
            )

            ax2.set_xlabel("Number of Supporting Targets")
            ax2.set_ylabel("Disease")

            plt.tight_layout()

            st.pyplot(fig2)

            # =====================
            # PUBMED VALIDATION
            # =====================

            st.header("4️⃣ PubMed Literature Validation")

            pubmed_rows = []

            for disease in top10["disease_name"].tolist():

                try:

                    evidence = pubmed_search(
                        drug.get("name", drug_input),
                        disease
                    )

                    pubmed_rows.append({
                        "Disease": disease,
                        "PubMed Results": evidence["count"]
                    })

                except Exception:

                    pubmed_rows.append({
                        "Disease": disease,
                        "PubMed Results": 0
                    })

            pubmed_df = pd.DataFrame(pubmed_rows)

            st.dataframe(
                pubmed_df,
                use_container_width=True,
                hide_index=True
            )

            # =====================
            # PUBMED GRAPH
            # =====================

            st.subheader("📚 Literature Evidence")

            fig3, ax3 = plt.subplots(figsize=(10, 5))

            ax3.barh(
                pubmed_df["Disease"].iloc[::-1],
                pubmed_df["PubMed Results"].iloc[::-1]
            )

            ax3.set_xlabel("Number of PubMed Results")
            ax3.set_ylabel("Disease")

            plt.tight_layout()

            st.pyplot(fig3)

            # =====================
            # NETWORK
            # =====================

            st.header("5️⃣ Drug–Target–Disease Network")

            G = nx.Graph()

            drug_name = drug.get("name", drug_input)

            G.add_node(
                drug_name,
                type="drug"
            )

            for target in targets:

                target_symbol = target.get("Target")

                if not target_symbol:
                    continue

                G.add_node(
                    target_symbol,
                    type="target"
                )

                G.add_edge(
                    drug_name,
                    target_symbol
                )

                related = ranking.head(5)

                for disease in related["disease_name"]:

                    G.add_node(
                        disease,
                        type="disease"
                    )

                    G.add_edge(
                        target_symbol,
                        disease
                    )

            fig4, ax4 = plt.subplots(figsize=(14, 9))

            pos = nx.spring_layout(
                G,
                seed=42
            )

            nx.draw(
                G,
                pos,
                ax=ax4,
                with_labels=True,
                node_size=1800,
                font_size=8
            )

            ax4.set_title(
                "Drug → Target → Disease Network"
            )

            st.pyplot(fig4)

            # =====================
            # SCIENTIFIC WARNING
            # =====================

            st.warning(
                "⚠️ Scientific note: This dashboard generates computational "
                "repurposing hypotheses. Open Targets association scores and "
                "PubMed co-occurrence do not establish clinical efficacy, "
                "causality, safety, or regulatory approval."
            )

else:

    st.info(
        "👈 Enter a drug name and click **Analyze Drug** to begin."
    )

    st.markdown(
        """
        ### 🔬 Workflow

        **Drug**
        ↓  
        **Molecular Targets**
        ↓  
        **Disease Associations**
        ↓  
        **AI Repurposing Score**
        ↓  
        **PubMed Validation**
        ↓  
        **Potential Repurposing Candidates**
        """
    )
