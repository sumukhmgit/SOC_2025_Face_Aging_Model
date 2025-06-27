people={"James":20, "Charles":34, "Rocky":40, "David":28, "Adam":30}
morethanthirty=[]
#Use a loop to iterate over dict.items().
for key, value in people.items():
#Use an if condition to filter names.
  if value>30:
    morethanthirty.append(key)
print(morethanthirty)