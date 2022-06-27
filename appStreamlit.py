import streamlit as st
# EDA Pkg
import pandas as pd
import joblib
import os
import sklearn
import numpy as np
from PIL import Image

# Data Viz Pkg
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')



# Create path to load all machine learning Models saved
def loading_savedModels(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Get Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key

# Get Value From Dictionary
def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value


def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)



def main():

	df = pd.read_csv("data/adult_salary.csv")
	df2 = pd.read_csv("data/adult_salary_data.csv")


	st.title('Predicting Salary')

	activity = ["Home", "EDA", "Prediction"]
	choice = st.sidebar.selectbox('Please choose an option', activity)

	# HomePage
	if choice == "Home":
		st.subheader("Write about what this is about")

	# EDA page
	if choice == "EDA":
		st.subheader("Below are the EDA options")

		if st.button('Show Column Names'):
			st.write(df.columns)

		if st.button('Show data size'):
			st.text("Number of Rows & Columns are:")
			st.write(df.shape)

		if st.button('Show detailed description about the data'):
			st.write(df.describe())

		if st.button("Target Class Totals"):
			st.write(df.iloc[:,-1].value_counts())

		# Allow user to see type of data
		if st.button("View sample of the data"):
			st.dataframe(df.head(15))

		if st.checkbox("View only selected columns of the data"):
				column_names = df.columns.tolist()
				view_columns = st.multiselect('Please choose the columns you would like to view',column_names)
				updated_df = df[view_columns]
				st.dataframe(updated_df.head(15))

		#st.subheader("Data Visualization")
		load_css('icon.css')
		load_icon('show_charts')
		# Show Correlation Plots with Matplotlib Plot
		if st.checkbox("View Correlation Plot"):
			st.write(sns.heatmap(df.corr(),annot=True))
			st.pyplot()




	# Prediction Page
	if choice == 'Prediction':
		st.subheader("Please input the values below:")

		# Dictionary of Mapped Values
		d_workclass = {"Never-worked": 0, "Private": 1, "Federal-gov": 2, "?": 3, "Self-emp-inc": 4, "State-gov": 5, "Local-gov": 6, "Without-pay": 7, "Self-emp-not-inc": 8}

		d_education = {"Some-college": 0, "10th": 1, "Doctorate": 2, "1st-4th": 3, "12th": 4, "Masters": 5, "5th-6th": 6, "9th": 7, "Preschool": 8, "HS-grad": 9, "Assoc-acdm": 10, "Bachelors": 11, "Prof-school": 12, "Assoc-voc": 13, "11th": 14, "7th-8th": 15}

		d_marital_status = {"Separated": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 2, "Married-civ-spouse": 3, "Never-married": 4, "Widowed": 5, "Divorced": 6}

		d_occupation = {"Tech-support": 0, "Farming-fishing": 1, "Prof-specialty": 2, "Sales": 3, "?": 4, "Transport-moving": 5, "Armed-Forces": 6, "Other-service": 7, "Handlers-cleaners": 8, "Exec-managerial": 9, "Adm-clerical": 10, "Craft-repair": 11, "Machine-op-inspct": 12, "Protective-serv": 13, "Priv-house-serv": 14}

		d_relationship = {"Other-relative": 0, "Not-in-family": 1, "Own-child": 2, "Wife": 3, "Husband": 4, "Unmarried": 5}

		d_race = {"Amer-Indian-Eskimo": 0, "Black": 1, "White": 2, "Asian-Pac-Islander": 3, "Other": 4}

		d_sex = {"Female": 0, "Male": 1}

		d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}

		d_class = {">50K": 0, "<=50K": 1}
		countries_img = {'af': 'Afghanistan','al': 'Albania','dz': 'Algeria','as': 'American Samoa','ad': 'Andorra','ao': 'Angola','ai': 'Anguilla','aq': 'Antarctica','ag': 'Antigua And Barbuda','ar': 'Argentina','am': 'Armenia','aw': 'Aruba','au': 'Australia','at': 'Austria','az': 'Azerbaijan','bs': 'Bahamas','bh': 'Bahrain','bd': 'Bangladesh','bb': 'Barbados','by': 'Belarus','be': 'Belgium','bz': 'Belize','bj': 'Benin','bm': 'Bermuda','bt': 'Bhutan','bo': 'Olivia','ba': 'Bosnia And Herzegovina','bw': 'Botswana','bv': 'Bouvet Island','br': 'Brazil','io': 'British Indian Ocean Territory','bn': 'Brunei Darussalam','bg': 'Bulgaria','bf': 'Burkina Faso','bi': 'Burundi','kh': 'Cambodia','cm': 'Cameroon','ca': 'Canada','cv': 'Cape Verde','ky': 'Cayman Islands','cf': 'Central African Republic','td': 'Chad','cl': 'Chile','cn': "People'S Republic Of China",'cx': 'Hristmas Island','cc': 'Cocos (Keeling) Islands','co': 'Colombia','km': 'Comoros','cg': 'Congo','cd': 'Congo, The Democratic Republic Of','ck': 'Cook Islands','cr': 'Costa Rica','ci': "Côte D'Ivoire",'hr': 'Croatia','cu': 'Cuba','cy': 'Cyprus','cz': 'Czech Republic','dk': 'Denmark','dj': 'Djibouti','dm': 'Dominica','do': 'Dominican Republic','ec': 'Ecuador','eg': 'Egypt','eh': 'Western Sahara','sv': 'El Salvador','gq': 'Equatorial Guinea','er': 'Eritrea','ee': 'Estonia','et': 'Ethiopia','fk': 'Falkland Islands (Malvinas)','fo': 'Aroe Islands','fj': 'Fiji','fi': 'Finland','fr': 'France','gf': 'French Guiana','pf': 'French Polynesia','tf': 'French Southern Territories','ga': 'Gabon','gm': 'Gambia','ge': 'Georgia','de': 'Germany','gh': 'Ghana','gi': 'Gibraltar','gr': 'Greece','gl': 'Greenland','gd': 'Grenada','gp': 'Guadeloupe','gu': 'Guam','gt': 'Guatemala','gn': 'Guinea','gw': 'Guinea-Bissau','gy': 'Guyana','ht': 'Haiti','hm': 'Heard Island And Mcdonald Islands','hn': 'Honduras','hk': 'Hong Kong','hu': 'Hungary','is': 'Iceland','in': 'India','id': 'Indonesia','ir': 'Iran, Islamic Republic Of','iq': 'Iraq','ie': 'Ireland','il': 'Israel','it': 'Italy','jm': 'Jamaica','jp': 'Japan','jo': 'Jordan','kz': 'Kazakhstan','ke': 'Kenya','ki': 'Kiribati','kp': "Korea, Democratic People'S Republic Of",'kr': 'Korea, Republic Of','kw': 'Kuwait','kg': 'Kyrgyzstan','la': "Lao People'S Democratic Republic",'lv': 'Latvia','lb': 'Lebanon','ls': 'Lesotho','lr': 'Liberia','ly': 'Libyan Arab Jamahiriya','li': 'Liechtenstein','lt': 'Lithuania','lu': 'Luxembourg','mo': 'Macao','mk': 'Macedonia, The Former Yugoslav Republic Of','mg': 'Madagascar','mw': 'Malawi','my': 'Malaysia','mv': 'Maldives','ml': 'Mali','mt': 'Malta','mh': 'Marshall Islands','mq': 'Martinique','mr': 'Mauritania','mu': 'Mauritius','yt': 'Mayotte','mx': 'Mexico','fm': 'Micronesia, Federated States Of','md': 'Moldova, Republic Of','mc': 'Monaco','mn': 'Mongolia','ms': 'Montserrat','ma': 'Morocco','mz': 'Mozambique','mm': 'Myanmar','na': 'Namibia','nr': 'Nauru','np': 'Nepal','nl': 'Netherlands','an': 'Netherlands Antilles','nc': 'New Caledonia','nz': 'New Zealand','ni': 'Nicaragua','ne': 'Niger','ng': 'Nigeria','nu': 'Niue','nf': 'Norfolk Island','mp': 'Northern Mariana Islands','no': 'Norway','om': 'Oman','pk': 'Pakistan','pw': 'Palau','ps': 'Palestinian Territory, Occupied','pa': 'Panama','pg': 'Papua New Guinea','py': 'Paraguay','pe': 'Peru','ph': 'Philippines','pn': 'Pitcairn','pl': 'Poland','pt': 'Portugal','pr': 'Puerto Rico','qa': 'Qatar','re': 'Réunion','ro': 'Romania','ru': 'Russian Federation','rw': 'Rwanda','sh': 'Saint Helena','kn': 'Saint Kitts And Nevis','lc': 'Saint Lucia','pm': 'Saint Pierre And Miquelon','vc': 'Saint Vincent And The Grenadines','ws': 'Samoa','sm': 'San Marino','st': 'Sao Tome And Principe','sa': 'Saudi Arabia','sn': 'Senegal','cs': 'Serbia And Montenegro','sc': 'Seychelles','sl': 'Sierra Leone','sg': 'Singapore','sk': 'Slovakia','si': 'Slovenia','sb': 'Solomon Islands','so': 'Somalia','za': 'South Africa','gs': 'South Georgia And South Sandwich Islands','es': 'Spain','lk': 'Sri Lanka','sd': 'Sudan','sr': 'Suriname','sj': 'Svalbard And Jan Mayen','sz': 'Swaziland','se': 'Sweden','ch': 'Switzerland','sy': 'Syrian Arab Republic','tw': 'Taiwan, Republic Of China','tj': 'Tajikistan','tz': 'Tanzania, United Republic Of','th': 'Thailand','tl': 'Timor-Leste','tg': 'Togo','tk': 'Tokelau','to': 'Tonga','tt': 'Trinidad And Tobago','tn': 'Tunisia','tr': 'Turkey','tm': 'Turkmenistan','tc': 'Turks And Caicos Islands','tv': 'Tuvalu','ug': 'Uganda','ua': 'Ukraine','ae': 'United Arab Emirates','gb': 'United Kingdom','us': 'United States','um': 'United States Minor Outlying Islands','uy': 'Uruguay','uz': 'Uzbekistan','ve': 'Venezuela','vu': 'Vanuatu','vn': 'Viet Nam','vg': 'British Virgin Islands','vi': 'U.S. Virgin Islands','wf': 'Wallis And Futuna','ye': 'Yemen','zw': 'Zimbabwe'}


		# Setup features for user to input 
		age = st.number_input("Please Enter Age between 60 & 90",16,90)
		
		workclass = st.selectbox("Please Choose Work Class",tuple(d_workclass.keys()))
		
		fnlwgt = st.number_input("Please Enter FNLWGT",1.228500e+04,1.484705e+06)
		
		education = st.selectbox("Please Enter Education Level",tuple(d_education.keys()))
		
		education_num = st.slider("Specify Education Number",1,16)
		
		marital_status = st.selectbox("Marital-status",tuple(d_marital_status.keys()))
		
		occupation = st.selectbox("Please Choose Occupation",tuple(d_occupation.keys()))
		
		relationship = st.selectbox("Please Choose Relationship",tuple(d_relationship.keys()))
		
		race = st.selectbox("Please Choose Race",tuple(d_race.keys()))
		
		sex = st.selectbox("Gender",tuple(d_sex.keys()))
		
		capital_gain = st.number_input("Please Enter Capital Gain between 0 & 99999",0,99999)
		
		capital_loss = st.number_input("Please Enter Capital Loss between 0 & 4356",0,4356)
		
		hours_per_week = st.number_input("Please Enter Number of Hours Worked Per Week ",0,99)
		
		native_country = st.selectbox("Please Select Country",tuple(d_native_country.keys()))

		# Using the get_value function, get the values of user input
		k_workclass = get_value(workclass,d_workclass)
		k_education = get_value(education,d_education)
		k_marital_status = get_value(marital_status,d_marital_status)
		k_occupation = get_value(occupation,d_occupation)
		k_relationship = get_value(relationship,d_relationship)
		k_race = get_value(race,d_race)
		k_sex = get_value(sex,d_sex)
		k_native_country = get_value(native_country,d_native_country)

		# Save the original input values and show to user the original and values
		user_inputed = [age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country]
		lis_user_inputed = [age ,k_workclass ,fnlwgt ,k_education ,education_num ,k_marital_status ,k_occupation ,k_relationship ,k_race ,k_sex ,capital_gain ,capital_loss ,hours_per_week ,k_native_country]
		sample_data = np.array(lis_user_inputed).reshape(1, -1)
		st.warning(user_inputed)
		st.success(lis_user_inputed)


		# Prediction section
		if st.checkbox("Make Prediction"):
			all_ml_dict = {'Logistic Regression':"LogisticRegression",
				'Random Forest':"RandomForestClassifier"}

			# Choose a model to use
			model_selection = st.selectbox('Please choose a model to use:',list(all_ml_dict.keys()))
			prediction_label = {"Greater Than 50K": 0, "Less Than 50K": 1}
			if st.button("Predict"):
				if model_selection == 'Random Forest':
					selected_model = loading_savedModels("models/salary_rf_model.pkl")
					make_prediction = selected_model.predict(sample_data)
				elif model_selection == 'Logistic Regression':
					selected_model = loading_savedModels("models/logistic_regression_salary.pkl")
					make_prediction = selected_model.predict(sample_data)
				
				final_result = get_key(make_prediction,prediction_label)
				st.success("The salary predicted is: {}".format(final_result))




if __name__ == '__main__':
	main()