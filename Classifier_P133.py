#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

#load the model from disk
import joblib
model = joblib.load(r"finalized_model.sav")
def apply_scalar_conversion(temp_df,predict_type):

    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()
    if 'churn' in temp_df.columns:
        temp_df = temp_df.drop(columns=["churn"])
    data = temp_df.drop(columns=['international_plan','voice_mail_plan'])
    column_features = data.columns.values
    if predict_type == 'Batch':
        scaled_data = scaler.fit_transform(data.to_numpy())
        result_df = pd.DataFrame(scaled_data)
    else:
        data = data.values.reshape(-1,1)
        scaled_data = scaler.fit_transform(data)
        scaled_data = scaled_data.reshape(1,-1)
        result_df = pd.DataFrame(scaled_data)


    result_df.columns = column_features
    result_df["international_plan"] = temp_df['international_plan']
    result_df["voice_mail_plan"] = temp_df['voice_mail_plan']
    return result_df
   

def main():
    #Setting Application title
    st.title('Telco Customer Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional telecommunication use case. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('app.jpg')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")

        st.subheader("Demographic data")
        account_length = st.number_input('The total no days that account is in active',min_value=1,value=122)
        day_calls		 = st.number_input('The total no of Day calls ',min_value=0,value=99)
        evening_calls		 = st.number_input('The total no of Evevnig calls ',min_value=0,value=89)
        night_calls		 = st.number_input('The total no of Night calls ',min_value=0,value=107)
        international_calls		 = st.number_input('The total no of International calls ',min_value=0,value=3)

        st.subheader("Payment data")
        totalcharges = st.number_input('The total amount charged to the customer', step=.1,format="%.2f",value=77.06)

        st.subheader("Services signed up for")
        international_plan = st.selectbox("Does the customer have International Plan", ('Yes','No'))
        voice_mail_plan = st.selectbox("Does the customer have Voice Mail Plan", ('Yes','No'))

        st.subheader("Customer Services")
        customer_service_calls = st.number_input('The total no of of Customer Service Calls',min_value=0,value=4)
        
        data = {
                'Account Length' :account_length,
                'Day Calls' :day_calls,
                'Evevning Calls' :evening_calls,
                'Night Calls' :night_calls,
                'International Calls': international_calls,
                'Customer Service Calls':customer_service_calls,
                'Total Charge': totalcharges,
                'International Plan': international_plan,
                'VoiceMail Plan':voice_mail_plan,
                
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        international_plan =  1 if international_plan == "Yes" else 0
        voice_mail_plan =  1 if voice_mail_plan == "Yes" else 0
        test_data = pd.DataFrame([{
                'account_length':account_length,
                'day_calls':day_calls,
                'evening_calls':evening_calls,
                'night_calls':night_calls,
                'international_calls':international_calls,
                'customer_service_calls':customer_service_calls,
                'total_charge':totalcharges,
                'international_plan':international_plan,
                'voice_mail_plan':voice_mail_plan,
       
        }])

        # #Preprocess inputs
        final_test_data = apply_scalar_conversion(test_data,"Online")
        prediction = model.predict(final_test_data)
       

        if st.button('Predict'):
           
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')
        

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        st.write("columns: {account_length, day_calls,evening_calls, night_calls, international_calls, customer_service_calls, total_charge, international_plan, voice_mail_plan}")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data)
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            if 'churn' in data.columns:
                data = data.drop(columns=["churn"])

            #Preprocess inputs
            final_test_data = apply_scalar_conversion(data,"Batch")

            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(final_test_data)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.', 
                                                    0:'No, the customer is happy with Telco Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

            
if __name__ == '__main__':
        main()