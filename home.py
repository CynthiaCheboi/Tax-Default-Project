import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px

from utils.data import read_data
from utils.data import *
from utils.plot import plot_model_results


model_path = 'models_metadata.pkl'
#note_book_path = 'Notebook.html'
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

        st.title("Tax Default Prediction Tool")

    
        st.markdown("### Problem Statement")

        st.markdown("""
        In today's dynamic tax landscape, ensuring tax compliance is crucial for sustaining government revenue collection. 
        However, a persistent challenge lies in the inability to predict and prevent instances of late or non-payment of taxes, 
        leading to an annual loss of Ksh. 20 billion. The current manual monitoring system lacks the capacity for early intervention, relying on reactive measures after taxpayers default. 
        With a staff member on average monitoring 1,000 taxpayers and limited resources for proactive engagement, 
        the need for a predictive model to identify default taxpayers in real-time is evident.
        """)

        st.markdown("### Objectives")

        st.markdown("""
        The primary goal of this predictive project is to develop a machine learning model that can accurately predict taxpayers that are likely to default
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
            net_profit = st.number_input("Net Profit")
        with col2:
            current_assets        = st.number_input("Current Assets")
        with col3:
            total_inventory           = st.number_input("Total Inventory")

        col4, col5,col6 = st.columns(3)
        with col4:
            proprtor_capt_reserves = st.number_input("Share Capital and Reserves")
        with col5:
            total_debt = st.number_input("Total Debt")  
        with col6:
            capital_allowance = st.number_input(label="Capital Allowance")

        col7, col8,col9 = st.columns(3)
        with col7:
            installment_tax_paid           = st.number_input("Installment Tax Paid")
        with col8:
            tax_payable_IT = st.number_input("IT Tax Payable")
        with col9:
            sales_exempt = st.number_input("Exempt Sales")
            
        col10, col11,col12 = st.columns(3)
        with col10:
            output_VAT = st.number_input("Output VAT")
        with col11:
            input_VAT = st.number_input("Input VAT")
            
        with col12:
            VAT_payable = st.number_input("VAT Payable")

        col13, col14,col15 = st.columns(3)
        with col13:
            less_crdt_bal_prv_mnth = st.number_input("Previous Month Credit Balance")
        with col14:
            WVAT_credit = st.number_input("WHVAT credit")
        with col15:
            VAT_advance_paid = st.number_input(label="VAT Advance Paid")

        col16, col17,col18 = st.columns(3)
        with col16:
            paye_payable           = st.number_input("[PAYE] payable")
        with col17:
            current_ratio = st.number_input(label="Current Ratio")
        with col18:
            quick_ratio = st.number_input(label="Quick Ratio")

        col19, col20 = st.columns(2)
        with col19:
            Gearing_ratio = st.number_input(label="Gearing Ratio")
        with col20:
            total_liabilities = st.number_input("Total Liabilities")

        submitted = st.form_submit_button(label='Predict')

    if submitted:
        # check if all paramaters have been checked
        list_of_params  = [net_profit, current_assets, total_inventory, proprtor_capt_reserves, total_debt,
                           VAT_payable, capital_allowance, installment_tax_paid, tax_payable_IT, sales_exempt,
                           output_VAT, input_VAT,VAT_payable, less_crdt_bal_prv_mnth, WVAT_credit, VAT_advance_paid,
                           paye_payable, current_ratio, quick_ratio, Gearing_ratio, total_liabilities]

        if list_of_params is not None:
            list_of_params = [float(i) for i in list_of_params]

            # create a dataframe
            data = pd.DataFrame(
                {
                    "net_profit": [net_profit],
                    "current_assets": [current_assets],
                    "total_inventory": [total_inventory],
                    "proprtor_capt_reserves": [proprtor_capt_reserves],
                    "total_debt": [total_debt],
                    "capital_allowance": [capital_allowance],
                    "installment_tax_paid": [installment_tax_paid],
                    "tax_payable_IT": [tax_payable_IT],
                    "sales_exempt": [sales_exempt],
                    "output_VAT": [output_VAT],
                    "input_VAT": [input_VAT],
                    "VAT_payable": [VAT_payable],
                    "less_crdt_bal_prv_mnth": [less_crdt_bal_prv_mnth],
                    "WVAT - credit": [WVAT_credit],
                    "VAT_advance_paid": [VAT_advance_paid],
                    "paye_payable": [paye_payable],
                    "current_ratio": [current_ratio],
                    "quick_ratio": [quick_ratio],
                    "Gearing_ratio": [Gearing_ratio],
                    "total_liabilities": [total_liabilities],
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