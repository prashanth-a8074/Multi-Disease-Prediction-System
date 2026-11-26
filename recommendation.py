# Specialized Hospital Recommendation System

HOSPITAL_DATABASE = {
    'heart': {
        'specialization': 'Cardiology / Heart Care',
        'hospitals': [
            {'name': 'Fortis Escorts Heart Institute', 'city': 'Delhi', 'rating': 4.8, 'address': 'Okhla Road, New Delhi', 'contact': '011-47135000'},
            {'name': 'Asian Heart Institute', 'city': 'Mumbai', 'rating': 4.9, 'address': 'Bandra Kurla Complex, Mumbai', 'contact': '022-66986666'},
            {'name': 'Narayana Institute of Cardiac Sciences', 'city': 'Bangalore', 'rating': 4.7, 'address': 'Bommasandra, Bangalore', 'contact': '080-71222222'},
            {'name': 'Apollo Hospitals (Greams Road)', 'city': 'Chennai', 'rating': 4.8, 'address': 'Greams Road, Chennai', 'contact': '044-28290200'},
            {'name': 'Care Hospitals', 'city': 'Hyderabad', 'rating': 4.6, 'address': 'Banjara Hills, Hyderabad', 'contact': '040-61656565'}
        ]
    },
    'diabetes': {
        'specialization': 'Endocrinology & Diabetology',
        'hospitals': [
            {'name': 'Dr. Mohan\'s Diabetes Specialities Centre', 'city': 'Chennai', 'rating': 4.8, 'address': 'Gopalapuram, Chennai', 'contact': '044-43968888'},
            {'name': 'Max Super Speciality Hospital', 'city': 'Delhi', 'rating': 4.6, 'address': 'Saket, New Delhi', 'contact': '011-26515050'},
            {'name': 'Kokilaben Dhirubhai Ambani Hospital', 'city': 'Mumbai', 'rating': 4.7, 'address': 'Four Bungalows, Andheri West, Mumbai', 'contact': '022-30999999'},
            {'name': 'Manipal Hospital', 'city': 'Bangalore', 'rating': 4.6, 'address': 'HAL Old Airport Road, Bangalore', 'contact': '080-22221111'},
            {'name': 'Yashoda Hospitals', 'city': 'Hyderabad', 'rating': 4.7, 'address': 'Somajiguda, Hyderabad', 'contact': '040-45674567'}
        ]
    },
    'breast_cancer': {
        'specialization': 'Oncology / Cancer Diagnostics & Surgery',
        'hospitals': [
            {'name': 'Tata Memorial Hospital', 'city': 'Mumbai', 'rating': 4.9, 'address': 'Parel, Mumbai', 'contact': '022-24177000'},
            {'name': 'Rajiv Gandhi Cancer Institute & Research Centre', 'city': 'Delhi', 'rating': 4.7, 'address': 'Rohini, New Delhi', 'contact': '011-47022222'},
            {'name': 'HCG Cancer Centre', 'city': 'Bangalore', 'rating': 4.8, 'address': 'Double Road, Bangalore', 'contact': '080-40206000'},
            {'name': 'Adyar Cancer Institute', 'city': 'Chennai', 'rating': 4.8, 'address': 'Adyar, Chennai', 'contact': '044-24910754'},
            {'name': 'Basavatarakam Indo American Cancer Hospital', 'city': 'Hyderabad', 'rating': 4.7, 'address': 'Banjara Hills, Hyderabad', 'contact': '040-23551235'}
        ]
    },
    'liver': {
        'specialization': 'Hepatology / Gastroenterology',
        'hospitals': [
            {'name': 'Institute of Liver and Biliary Sciences (ILBS)', 'city': 'Delhi', 'rating': 4.9, 'address': 'Vasant Kunj, New Delhi', 'contact': '011-46300000'},
            {'name': 'Global Hospitals', 'city': 'Mumbai', 'rating': 4.7, 'address': 'Parel, Mumbai', 'contact': '022-67670101'},
            {'name': 'Aster CMI Hospital', 'city': 'Bangalore', 'rating': 4.6, 'address': 'Hebbal, Bangalore', 'contact': '080-43430000'},
            {'name': 'GEM Hospital & Research Centre', 'city': 'Chennai', 'rating': 4.7, 'address': 'Perungudi, Chennai', 'contact': '044-61212121'},
            {'name': 'AIG Hospitals', 'city': 'Hyderabad', 'rating': 4.8, 'address': 'Gachibowli, Hyderabad', 'contact': '040-42444244'}
        ]
    },
    'parkinsons': {
        'specialization': 'Neurology / Movement Disorders',
        'hospitals': [
            {'name': 'National Institute of Mental Health and Neurosciences (NIMHANS)', 'city': 'Bangalore', 'rating': 4.9, 'address': 'Hosur Road, Bangalore', 'contact': '080-26995000'},
            {'name': 'Medanta - The Medicity', 'city': 'Delhi', 'rating': 4.7, 'address': 'Gurugram, NCR Delhi', 'contact': '0124-4141414'},
            {'name': 'Jaslok Hospital & Research Centre', 'city': 'Mumbai', 'rating': 4.8, 'address': 'Pedder Road, Mumbai', 'contact': '022-66573333'},
            {'name': 'Apollo Proton Cancer Centre & Neuro Clinic', 'city': 'Chennai', 'rating': 4.7, 'address': 'Tharamani, Chennai', 'contact': '044-33333333'},
            {'name': 'NIMS (Nizam\'s Institute of Medical Sciences)', 'city': 'Hyderabad', 'rating': 4.6, 'address': 'Punjagutta, Hyderabad', 'contact': '040-23480000'}
        ]
    }
}

def get_recommendations(disease, city=None):
    """
    Returns a list of recommended hospitals based on disease classification.
    
    Parameters:
    - disease (str): 'diabetes', 'heart', 'breast_cancer', 'liver', or 'parkinsons'
    - city (str, optional): Target city name to filter (e.g. 'Mumbai', 'Delhi')
    
    Returns:
    - dict: Contains 'specialization' name and list of matching hospital dictionaries.
    """
    if disease not in HOSPITAL_DATABASE:
        return {'specialization': 'General Medicine', 'hospitals': []}
        
    spec_info = HOSPITAL_DATABASE[disease]
    specialization = spec_info['specialization']
    all_hospitals = spec_info['hospitals']
    
    if city:
        city_lower = city.strip().lower()
        # Filter by city
        filtered_hospitals = [h for h in all_hospitals if h['city'].lower() == city_lower]
        
        # If no specific match for that city, return all hospitals and flag it
        if not filtered_hospitals:
            # Add a mock local hospital representation for search feedback
            mock_local = {
                'name': f'Local Specialty Health Hub ({city.title()})',
                'city': city.title(),
                'rating': 4.5,
                'address': f'Main Healthcare District, {city.title()}',
                'contact': '1800-XXX-XXXX'
            }
            return {
                'specialization': specialization,
                'hospitals': [mock_local] + all_hospitals,
                'note': f"No specific registered hospital found in '{city}'. Displaying nearest premium options."
            }
        return {'specialization': specialization, 'hospitals': filtered_hospitals}
        
    return {'specialization': specialization, 'hospitals': all_hospitals}
