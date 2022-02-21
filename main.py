import pickle

"""
test for saving the scaler
"""

mylist = [1,2,3,4,5,6,7]

with open('./saved_scalers/parrot.pkl', 'wb') as f:
    pickle.dump(mylist, f)

with open('./saved_scalers/parrot.pkl', 'rb') as f:
    mynewlist = pickle.load(f)

print(mynewlist)