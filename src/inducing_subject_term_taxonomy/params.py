
class Param(object):

  def __init__(self, path='params.dat'):
    with open(path, 'r') as fo:
      for line in fo.readlines():
        if len(line) < 2: continue
        if (line.startswith("#")): continue

        parts = [s.strip(' :\n') for s in line.split(' ', 1)]
        self.__dict__[parts[0]] = parts[1]
        fo.close()