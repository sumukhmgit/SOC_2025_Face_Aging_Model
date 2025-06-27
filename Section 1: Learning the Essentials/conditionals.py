#Use if, elif, else to handle the conditions.
def categorize_age(age):
    # Write your logic here
    if age<13:
      return 'Child'
    elif age<20:
      return 'Teen'
    elif age<60:
      return 'Adult'
    else:
      return 'Senior'