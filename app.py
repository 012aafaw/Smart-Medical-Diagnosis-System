import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import base64

# تحسينات واجهة المستخدم
def set_bg_hack(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover;
            background-attachment: fixed;
        }}
        .css-1aumxhk {{
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
        }}
        .st-bb {{ background-color: rgba(255, 255, 255, 0.8); }}
        .st-at {{ background-color: #f0f2f6; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# تحميل البيانات
@st.cache_data
def load_data():
    train_df = pd.read_csv('Training.csv')
    test_df = pd.read_csv('Testing.csv')
    return train_df, test_df

try:
    train_df, test_df = load_data()
except FileNotFoundError:
    st.error("الرجاء التأكد من وجود ملفات البيانات (Training.csv و Testing.csv) في المسار الصحيح")
    st.stop()

# تنظيف البيانات
train_df = train_df.dropna(axis=1)
test_df = test_df.dropna(axis=1)

# تحضير البيانات
X_train = train_df.iloc[:, :-1]
y_train = train_df['prognosis']

# إنشاء وتدريب النموذج
@st.cache_resource
def train_model():
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)
    
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
    model.fit(X_train, y_encoded)
    
    return model, encoder

model, encoder = train_model()

# واجهة المستخدم المحسنة
set_bg_hack('medical_bg.png')  # احفظ صورة خلفية في نفس المجلد

st.title('🏥 نظام التشخيص الطبي الذكي')
st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">اختر أعراضك وأدخل معلوماتك للحصول على تشخيص أولي</p>', unsafe_allow_html=True)

# قسم معلومات المريض
with st.expander("🔍 المعلومات الشخصية", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("العمر", 1, 100, 30)
    with col2:
        gender = st.radio("الجنس", ['ذكر', 'أنثى'], horizontal=True)

# قسم اختيار الأعراض
st.subheader("🧪 اختر الأعراض التي تعاني منها")
st.write("يمكنك اختيار عدة أعراض من القوائم التالية:")

# إنشاء تبويبات للأعراض حسب التصنيف
tab1, tab2, tab3 = st.tabs(["أعراض عامة", "أعراض موضعية", "أعراض أخرى"])

all_symptoms = X_train.columns.tolist()
general_symptoms = all_symptoms[:40]
local_symptoms = all_symptoms[40:80]
other_symptoms = all_symptoms[80:]

symptom_selection = {}

with tab1:
    cols = st.columns(2)
    for i, symptom in enumerate(general_symptoms):
        with cols[i%2]:
            display_name = symptom.replace('_', ' ').title()
            symptom_selection[symptom] = st.checkbox(display_name, key=symptom)

with tab2:
    cols = st.columns(2)
    for i, symptom in enumerate(local_symptoms):
        with cols[i%2]:
            display_name = symptom.replace('_', ' ').title()
            symptom_selection[symptom] = st.checkbox(display_name, key=symptom)

with tab3:
    cols = st.columns(2)
    for i, symptom in enumerate(other_symptoms):
        with cols[i%2]:
            display_name = symptom.replace('_', ' ').title()
            symptom_selection[symptom] = st.checkbox(display_name, key=symptom)

# زر التنبؤ مع تأثيرات بصرية
if st.button('🔍 الحصول على التشخيص', use_container_width=True):
    # تحضير بيانات الإدخال
    input_data = pd.DataFrame(columns=X_train.columns)
    input_data.loc[0] = 0
    
    # تعيين القيم 1 للأعراض المختارة
    selected_symptoms = [s for s, sel in symptom_selection.items() if sel]
    for symptom in selected_symptoms:
        input_data[symptom] = 1
    
    if not selected_symptoms:
        st.error("⚠ الرجاء اختيار عرض واحد على الأقل")
    else:
        with st.spinner('جاري التحليل...'):
            # التنبؤ
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            
            # الحصول على أفضل 3 تنبؤات
            top3 = np.argsort(probability[0])[-3:][::-1]
            
            st.success("✅ تم التحليل بنجاح")
            st.balloons()
            
            # عرض النتائج في بطاقات
            st.subheader("📋 النتائج:")
            
            for i, idx in enumerate(top3):
                disease = encoder.classes_[idx]
                prob = probability[0][idx] * 100
                
                with st.container():
                    st.markdown(f"""
                    <div style="
                        padding: 15px;
                        border-radius: 10px;
                        margin: 10px 0;
                        background-color: #f0f8ff;
                        border-left: 5px solid #4b8df8;
                    ">
                        <h4>{i+1}. {disease}</h4>
                        <p>احتمالية: <strong>{prob:.2f}%</strong></p>
                        <progress value="{prob}" max="100" style="width:100%; height:20px;"></progress>
                        <p><small>العمر والجنس: {age} سنة، {gender}</small></p>
                    </div>
                    """, unsafe_allow_html=True)

# معلومات جانبية محسنة
st.sidebar.image("medical_logo.png", width=200)  # احفظ صورة شعار
st.sidebar.title("ℹ معلومات النظام")
st.sidebar.write("""
هذا النظام يستخدم خوارزميات الذكاء الاصطناعي للتنبؤ بالأمراض بناءً على الأعراض المدخلة.

*ملاحظة هامة:*  
هذه النتائج هي لأغراض استشارية أولية فقط ولا تغني عن استشارة الطبيب المختص.
""")

st.sidebar.header("📊 إحصائيات")
st.sidebar.write(f"✅ عدد الأمراض المدربة: {len(encoder.classes_)}")
st.sidebar.write(f"📌 عدد الأعراض المتاحة: {len(all_symptoms)}")
st.sidebar.write(f"🧑‍⚕ معلومات المريض: {age} سنة، {gender}")


# إعدادات للنشر
st.sidebar.header("⚙ الإعدادات")
debug_mode = st.sidebar.checkbox("وضع التصحيح")
if debug_mode:

    st.sidebar.write("المتغيرات المختارة:", selected_symptoms if 'selected_symptoms' in locals() else [])
