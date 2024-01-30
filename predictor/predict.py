def predict_gen(meta1):
    import pickle
    import os
    from django.conf import settings
    path = os.path.join(settings.MODELS, 'model3.p')
    with open(path, 'rb') as pickled:
        data = pickle.load(pickled)
    model = data['model']
    norma = data['norma']
    lgn = data['lgn']
    x = norma.transform([meta1])
    pred = model.predict(x)
    return(lgn[pred[0]])

