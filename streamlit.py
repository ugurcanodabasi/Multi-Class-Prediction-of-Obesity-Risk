if main_tab == "Ana Sayfa":
    left_col, right_col = main_tab.columns(2)

    left_col.write("""Bu projenin amacı, bireylerde kardiyovasküler hastalıklarla ilişkili obezite riskini tahmin etmek için çeşitli faktörleri kullanmaktır. Kardiyovasküler hastalıklar, dünya genelinde sağlık sorunlarının önde gelen nedenlerinden biri olarak kabul edilmektedir. Bu hastalıkların birçoğu obezite ile doğrudan ilişkilidir. Bu nedenle, obeziteyi öngörmek ve bu konuda farkındalık yaratmak önemlidir.""")

    left_col.write("""Veri Seti ve Hedef
    Bu projede kullanılan veri seti, bireylerin demografik bilgilerini, yaşam tarzı alışkanlıklarını ve fizyolojik ölçümlerini içerir. Ölçümler arasında boy, kilo, günlük su tüketimi, fiziksel aktivite düzeyi gibi faktörler bulunmaktadır. Veri setindeki her bir satır, bir bireyi temsil eder ve bu bireylerin obezite durumları "NObeyesdad" sütununda belirtilmiştir.""")

    #TAVSİYE:Veri setinin bir kısmı eklenebilir

    right_col.write("""Kullanılan Algoritmalar
    Bu proje, LightGBM makine öğrenimi modeli kullanmaktadır. LightGBM, yüksek performanslı ve hızlı bir gradyan arttırma (gradient boosting) algoritmasıdır. Bu algoritma, veri setindeki örüntüleri öğrenerek ve karmaşık ilişkileri modelleyerek obezite riskini tahmin etmek için kullanılır.""")

    right_col.write("""Uygulama: Streamlit ile Model Tahmini
    Bu projede, geliştirilen modelin kullanıcı dostu bir arayüz ile sunulması amaçlanmıştır. Streamlit adlı Python kütüphanesi, basit ve etkileşimli web uygulamaları oluşturmayı sağlar. Bu projede, geliştirilen LightGBM modeli Streamlit arayüzü ile entegre edilmiştir.
    Kullanıcılar, arayüz üzerinden bireysel özellikleri girebilir ve modele besleyerek obezite risk tahminini alabilirler. Bu tahminler, bireylerin normal kilolu, aşırı kilolu, obez veya aşırı obez olma riskini belirtir.""")

if chart_tab == "Grafikler":
    col1, col2 = chart_tab.columns(2)

    with col1:
        st.header("Korelasyon Matrisi")
        st.image("korelasyon.png")

    with col2:
        st.header("Shap")
        st.image("SHAP.png")


#Tahmin ########################################################

if prediction_tab == "Model":

    if st.session_state['tahmin']:
        model_cont.subheader("Tahmin")
        col1, col2, col3 = model_cont.columns(3)
        selected_age = col1.number_input("Yaş")
        selected_gender = col2.number_input("Cinsiyet")
        selected_weight = col3.number_input("Kilo")
        selected_height = col4.number_input("Boy")
        selected_CH2O = col5.number_input("Günlük su tüketimi")

        if col6.button("Tahminle"):
            prediction = predict_model(df, selected_age, selected_gender, selected_weight, selected_height,
                                       selected_CH2O)
            col6.metric(label="Tahmin Edilen Obezite Riski", value=(prediction[0]))
            st.balloons()

