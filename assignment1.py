def build_graph():
  with open("edmonton-roads-2.0.1.txt") as file: # Assumes filename is edmonton-roads-2.0.1.txt
    for line in file: # Variable 'line' loops over each line in the file
      line = line.strip().split(',') # Remove trailing newline character and splits line into list
      if line[0] == 'V':
        
