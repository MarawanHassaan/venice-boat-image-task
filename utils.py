def binary_encoding(data):
    from sklearn.preprocessing import LabelBinarizer
    return LabelBinarizer().fit_transform(data)


def integer_encoding(data):
    from sklearn.preprocessing import LabelEncoder
    return LabelEncoder().fit_transform(data)
