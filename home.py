import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

from utils.data import read_data
from utils.data import *
from utils.plot import plot_model_results


model_path = 'models_metadata.pkl'
#note_book_path = 
import streamlit.components.v1 as components


# Set Page config

st.set_page_config(
    page_title="Tax Default Prediction Tool", 
    page_icon="💵", 
    layout="wide"
)

# menu bar
with st.sidebar:
    selected = option_menu(None, ["Home", "EDA", "Models", "Prediction","Interpretation"],
    icons     =['house', 'cloud-upload', "list-task", 'gear'],
    menu_icon ="cast", 
    default_index=0, 
    orientation="vertical",

    styles={
    "icon"              : {"color": "orange", "font-size": "16px"},
    "nav-link"          : {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
    "nav-link-selected" : {"background-color": "blue"},
    })


############################## Home Page  #################################################    
###########################################################################################
def main():
    if selected == "Home":
       
        # Logo section
        st.image('kra-logo.png', width=200)

        st.title("Tax Default Prediction Tool")

    
        st.markdown("### Problem Statement")

        st.markdown("""
        In today's dynamic tax landscape, ensuring tax compliance is crucial for sustaining government revenue collection. 
        However, a persistent challenge lies in the inability to predict and prevent instances of late or non-payment of taxes, 
        leading to an annual loss of Ksh. 20 billion. The current manual monitoring system lacks the capacity for early intervention, relying on reactive measures after taxpayers default. 
        With an average staff member monitoring 1,000 taxpayers and limited resources for proactive engagement, 
        the need for a predictive model to identify high-risk taxpayers in real-time is evident.
        """)

        st.markdown("### Objectives")

        st.markdown("""
        The primary goal of this predictive project is to develop a machine learning model that can accurately predict risky taxpayers 
        in real-time using historical data. Specifically, the following objectives are explored:
        
        - **Identify Key Predictive Features:** Analyze taxpayer demographics and past declaration information to determine the most important features for predicting tax evasion.
        
        - **Apply, Test, and Evaluate a Machine Learning Model:** Build a predictive model to assess the likelihood of taxpayers defaulting. This will facilitate case selection, noting that tax administrations have limited resources and therefore cannot address all compliance risks.
        
        - **Optimize Resource Allocation:** Utilizing the results to implement targeted interventions such as enforcement actions, tax audits, and taxpayer education, which are imperative to minimize revenue loss.
        """)            
        

if __name__ == "__main__":
    main()

   

############################## EDA Page  #################################################    
###########################################################################################
if selected == 'EDA':
    st.header(":orange[Tax Default Prediction Tool Exploratory Analysis]",divider=True)
    

    renderer = get_pyg_renderer()
    renderer.render_explore()

############################## Models Page  #################################################    
###########################################################################################
if selected == "Models":
    st.subheader(":orange[Trained Models Information]",divider=True)

    with st.expander('Training Data'):
        st.dataframe(read_data())
    loaded_models , loaded_model_results = model_load()
    df = pd.DataFrame(loaded_model_results)
    col9, col10, col11, col12 = st.columns(4)
    with col9:
       with st.container(border=True):
            st.plotly_chart(plot_model_results(df,'accuracy_score'),use_container_width=True)
    with col10:
       with st.container(border=True):
            st.plotly_chart(plot_model_results(df,'f1_score'),use_container_width=True)
    with col11:
       with st.container(border=True):
            st.plotly_chart(plot_model_results(df,'precision_score'),use_container_width=True)
    with col12:
       with st.container(border=True):
            st.plotly_chart(plot_model_results(df,'recall_score'),use_container_width=True)

#with st.expander("Learn how the model was trained?"):

#        with open(note_book_path, 'r',encoding='utf-8') as f:
#            html_data = f.read()
#        components.html(html_data, height=1000, width=800, scrolling=True)

# Download Model
st.sidebar.markdown("### Download")
download_choice=st.sidebar.selectbox(label='Select model to download 👇',options=["Serialized Model"])

if download_choice=='Model':
    download_objects(model_path)
#if download_choice=='Notebook':
#    download_objects(note_book_path)
        

############################## Prediction Page  #################################################    
###########################################################################################
if selected == "Prediction":

    st.header(":orange[Prediction Page]",divider=True)
    st.markdown(" ")

    with st.form('my_form',border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            Effective_tax_rate       = st.number_input("Effective Tax Rate")
        with col2:
            VAT_payable           = st.number_input("VAT payable")
        with col3:
            installment_tax_paid           = st.number_input("Installment Tax Paid")

        col4, col5,col6 = st.columns(3)
        with col4:
            WVAT_credit = st.number_input("WHVAT credit")
        with col5:
            output_VAT = st.number_input("Output VAT")
        with col6:
            VAT_input_output = st.number_input("VAT Input Output")

        col7, col8,col9 = st.columns(3)
        with col7:
            current_assets        = st.number_input("Current Assets")
        with col8:
            total_sales = st.number_input("Total Sales")
        with col9:
            less_crdt_bal_prv_mnth = st.number_input("Previous Month Credit Balance")

        col10, col11,col12 = st.columns(3)
        with col10:
            input_VAT = st.number_input("Input VAT")
        with col11:
            total_liabilities = st.number_input("Total Liabilities")
        with col12:
            total_expenses = st.number_input("Total Expenses")

        col13, col14,col15 = st.columns(3)
        with col13:
            total_purchases = st.number_input("Total Purchases")
        with col14:
            current_liabilities = st.number_input("Current Liabilities")
        with col15:
            profit_margin = st.number_input("Profit Margin")

        col16, col17,col18 = st.columns(3)
        with col16:
            proprtor_capt_reserves = st.number_input("Share Capital and Reserves")
        with col17:
            paye_payable           = st.number_input("[PAYE] payable")
        with col18:
            gross_turnover = st.number_input("Gross Turnover")

        col19, col20 = st.columns(2)
        with col19:
            Age = st.number_input("Age")
        with col20:
            net_profit = st.number_input("Net Profit")

        submited = st.form_submit_button(label='Predict')

    if submited:
        # check if all paramaters have been checked
        list_of_params  = [Effective_tax_rate, VAT_payable, installment_tax_paid, WVAT_credit, output_VAT,
                           VAT_input_output, current_assets, total_sales, less_crdt_bal_prv_mnth, input_VAT,
                           total_liabilities, total_expenses, total_purchases, current_liabilities, profit_margin,
                            proprtor_capt_reserves, paye_payable, gross_turnover, Age, net_profit ]

        if list_of_params is not None:
            list_of_params = [float(i) for i in list_of_params]

            # create a dataframe
            data = pd.DataFrame(
                {
                    "Effective_tax_rate": [Effective_tax_rate],
                    "VAT_payable": [VAT_payable],
                    "installment_tax_paid": [installment_tax_paid],
                    "WVAT - credit": [WVAT_credit],
                    "output_VAT": [output_VAT],
                    "VAT_input_output": [VAT_input_output],
                    "current_assets": [current_assets],
                    "total_sales": [total_sales],
                    "less_crdt_bal_prv_mnth": [less_crdt_bal_prv_mnth],
                    "input_VAT": [input_VAT],
                    "total_liabilities": [total_liabilities],
                    "total_expenses": [total_expenses],
                    "total_purchases": [total_purchases],
                    "current_liabilities": [current_liabilities],
                    "profit_margin": [profit_margin],
                    "proprtor_capt_reserves": [proprtor_capt_reserves],
                    "paye_payable": [paye_payable],
                    "gross_turnover": [gross_turnover],
                    "Age": [Age],
                    "net_profit": [net_profit],

                })
            # st.dataframe(data)

            # st.write(predict_model(data))
            st.markdown("#### Predictions Results")
            col11, col12, col13, col14, col15 = st.columns(5)

            predict_results = predict_model(data)

            col11.warning(list(predict_results.keys())[0])
            col11.success(model_category_using_y_preds(predict_results['AdaBoostClassifier']['prediction'][0]))
            col11.markdown(':grey[Probability of Late payment] ')
            col11.info(round((predict_results['AdaBoostClassifier']['probability'])* 100,2))

            col12.warning(list(predict_results.keys())[1])
            col12.success(model_category_using_y_preds(predict_results['XGBClassifier']['prediction'][0]))
            col12.markdown(':grey[Probability of Late payment] ')
            col12.info(round((predict_results['XGBClassifier']['probability'])* 100,2))

            col13.warning(list(predict_results.keys())[2])
            col13.success(model_category_using_y_preds(predict_results['RandomForestClassifier']['prediction'][0]))
            col13.markdown(':grey[Probability of Late payment] ')
            col13.info(round((predict_results['RandomForestClassifier']['probability'])* 100,2))

            col14.warning(list(predict_results.keys())[3])
            col14.success(model_category_using_y_preds(predict_results['DecisionTreeClassifier']['prediction'][0]))
            col14.markdown(':grey[Probability of Late payment] ')
            col14.info(round((predict_results['DecisionTreeClassifier']['probability'])* 100,2))

            col15.warning(list(predict_results.keys())[4])
            col15.success(model_category_using_y_preds(predict_results['LogisticRegression']['prediction'][0]))
            col15.markdown(':grey[Probability of Late payment] ')
            col15.info(round((predict_results['LogisticRegression']['probability'])* 100,2))
        else:
            st.warning('Please fill all the fields')

    else:
        st.warning("Please fill the form to Predict the chance of Late payment")


############################## Interpretation Page  #################################################    
###########################################################################################
if selected == 'Interpretation':
    data   = read_data()
    st.markdown("### :orange[Explainable AI]")

    data_instance = st.sidebar.selectbox("Select a Data Instance",options=data.index.to_list())
    st.data_editor(data,use_container_width=-True,height=250)
    st.markdown('👈Please select Data Instance')

    if data_instance:
        data_picked=data.loc[[data_instance]]
        st.write('Data Instance Selected')
        st.data_editor(data_picked, use_container_width=True)

        on = st.toggle("Show Interpretability")
        if on:
            with st.container(border=True):
                components.html(lime_explainer(read_data(),12), height=800, width=900, scrolling=True)