import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

# Chargement du modèle
model = pickle.load(open('model.pkl', 'rb'))

def main():
    st.title("Prédiction de Décision d'Embauche")

    with st.form("hiring_form"):
        st.write("Veuillez entrer les détails du candidat:")

        education = st.selectbox("Niveau d'Éducation", ['Bachelor', 'License', 'Master', 'PhD'])
        personality_score = st.slider("Score de Personnalité", min_value=0, max_value=100, step=1)
        skills = st.slider("Score de Compétences", min_value=0, max_value=100, step=1)
        interview_score = st.slider("Score d'Entretien", min_value=0, max_value=100, step=1)

        submitted = st.form_submit_button("Prédire")

        if submitted:
            input_data = pd.DataFrame({
                'EducationLevel': [education],
                'SkillScore': [skills],
                'InterviewScore': [interview_score],
                'PersonalityScore': [personality_score]
            })

            ordinal_encoder = OrdinalEncoder(categories=[['Bachelor', 'License', 'Master', 'PhD']])
            input_data[['EducationLevel']] = ordinal_encoder.fit_transform(input_data[['EducationLevel']])

            prediction = model.predict(input_data)[0]

            
            colored_box = f'''
                <div style="background-color: {'#88ff88' if prediction else '#ff8888'}; padding: 8px; border-radius: 25px; text-align: center; max-width: 250px; max-height: 150px; margin: 10px auto;">
                    <p style="color: black; font-size: 20px; font-weight: bold;">Décision d'Embauche : {'Accepté ✅' if prediction else 'Refusé ❌'}</p>
                </div>
            '''
            st.markdown(colored_box, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
