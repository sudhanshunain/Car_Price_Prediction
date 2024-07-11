from django.shortcuts import render
import joblib
import pandas as pd

model = joblib.load('./savedmodels/model.joblib')
loc_encoder = joblib.load('./savedmodels/loc_encoder.joblib')
owner_encoder = joblib.load('./savedmodels/owner_encoder.joblib')
fuel_encoder = joblib.load('./savedmodels/fuel_encoder.joblib')
brand_encoder = joblib.load('./savedmodels/brand_encoder.joblib')
scale = joblib.load('./savedmodels/scale.joblib')
transmission_encoder = joblib.load('./savedmodels/transmission_encoder.joblib')

def predictor(request):
    return render(request, 'main.html')

def formInfo(request):
    df = pd.DataFrame()

    df = pd.DataFrame({
        'Location': [request.GET.get('Location')],
        'Year': [request.GET.get('Year')],
        'Brand': [request.GET.get('Brand')],
        'Fuel_Type': [request.GET.get('Fuel_Type')],
        'Engine': [request.GET.get('Engine')],
        'Power': [request.GET.get('Power')],
        'Mileage': [request.GET.get('Mileage')],
        'Seats': [request.GET.get('Seats')],
        'Transmission': [request.GET.get('Transmission')],
        'Owner_Type': [request.GET.get('Owner_Type')]
    })

    df['Location_cat'] = loc_encoder.transform(df[['Location']])
    df['Brand_cat'] = brand_encoder.transform(df[['Brand']])
    df['Fuel_Type_cat'] = fuel_encoder.transform(df[['Fuel_Type']])
    df['Transmission_cat'] = transmission_encoder.transform(df[['Transmission']])
    df['Owner_Type_cat'] = owner_encoder.transform(df[['Owner_Type']])
    df[['Engine','Power', 'Mileage', 'Year']] = scale.transform(df[['Engine','Power', 'Mileage', 'Year']])

    X = df[['Location_cat', 'Fuel_Type_cat', 'Transmission_cat', 'Owner_Type_cat', 'Brand_cat', 
            'Year', 'Engine', 'Power', 'Seats','Mileage']] 
    Y= model.predict(X) 
    print(Y) 
    return render(request, 'result.html', {'result': Y[0] })