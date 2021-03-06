import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def month2num (month):

    if month == "Jan":
        return 0

    elif month == "Feb":
        return 1

    elif month == "Mar":
        return 2

    elif month == "Apr":
        return 3

    elif month == "May":
        return 4

    elif month == "June":
        return 5

    elif month == "Jul":
        return 6

    elif month == "Aug":
        return 7

    elif month == "Sep":
        return 8

    elif month == "Oct":
        return 9

    elif month == "Nov":
        return 10

    else:
        return 11
    


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # load data just like the demo in lecture
    with open(filename) as f:

        # read data into
        reader = csv.reader(f)
        # skip the header in csv
        next(reader)

        evidence = []
        labels = []


        # loop through the reader
        for row in reader:
            evilist = []
            for cell in row[:17]:
                if cell is row[10]:
                    evilist.append(month2num(cell))
                elif cell is row[15]:
                    if cell == "Returning_Visitor":
                        evilist.append("1")
                    else:
                        evilist.append("0")
                elif cell is row[16]:
                    if cell == "TRUE":
                        evilist.append("1")
                    else:
                        evilist.append("0")
                else:
                    evilist.append(cell)
            
            evidence.append(evilist)
            labels.append(["1" if row[17] == "TRUE" else "0"])
    
    return (evidence,labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # initiate k-nearest neighbor model where k = 1
    model = KNeighborsClassifier(n_neighbors=1)

    # fit models with training set data
    model.fit(evidence, labels)

    # need to match the formate of list of list
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    tot_pos = 0
    tot_neg = 0
    act_pos = 0
    act_neg = 0

    for actual,predicted in zip(labels,predictions):
        if actual[0] == "1":
            tot_pos += 1
            if actual[0] == predicted:
                act_pos += 1
        else:
            tot_neg += 1
            if actual[0] == predicted:
                act_neg += 1
    
    sensitivity = float(act_pos/tot_pos)
    specificity = float(act_neg/tot_neg)

    return (sensitivity,specificity)


if __name__ == "__main__":
    main()
