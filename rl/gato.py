class JuegoGato:
  #Comienza el raton, valor=-1
  def __init__(self,estado=[0]*9,turno=-1):
    self.tablero=estado
    self.completo=False
    self.ganador=None
    self.jugador=turno

  def reiniciar(self):
    self.tablero=[0]*9
    self.completo=False
    self.ganador=None
    self.jugador=-1

  def get_jugador(self):
    return "O" if self.jugador==-1 else "X"

  def get_estado(self):
    estado=""
    for i in range(9):
      if self.tablero[i]==-1:
        estado+="O"
      elif self.tablero[i]==1:
        estado+="X"
      else:
        estado+=" "
    return estado

  def generar_jugadas_posibles(self):
    posibles=[]
    for i in range(9):
      if self.tablero[i]==0:
        posibles.append(i)
    return posibles

  def estado_final(self):
    self.evaluar()
    if self.ganador is not None or self.completo:
      return True
    else:
      return False

  def evaluar(self):
    if 0 not in self.tablero:
      self.completo=True
    else:
      self.completo=False
    estado=[]
    for i in [0,3,6]:
      estado.append(sum(self.tablero[i:i+3]))
    for i in [0,1,2]:
      estado.append(self.tablero[i]+self.tablero[i+3]+self.tablero[i+6])
    estado.append(self.tablero[0]+self.tablero[4]+self.tablero[8])
    estado.append(self.tablero[2]+self.tablero[4]+self.tablero[6])
    for valor in estado:
      if valor==3 or valor==-3:
        self.ganador=valor//3
        return
    if self.completo:
      self.ganador=0
    else:
      self.ganador=None

  def calcular_utilidad(self):
    return self.ganador

  def jugar(self,jugada):
    self.tablero[jugada]=self.jugador
    self.jugador*=-1

  def deshacer_jugada(self,jugada):
    self.tablero[jugada]=0
    self.jugador*=-1

#-1: Ratón (Inicia, es el jugador humano)
# 1: Gato (Responde, es el computador)
# cuando gana el gato el valor es 1
# cuando gana el ratón el valor es -1
# un empate tiene utilidad 0
# etapa  1: maximizar
# etapa -1: minimizar
def minimax(juego,etapa,secuencia,secuencias):
  if juego.estado_final():
    utilidad=juego.calcular_utilidad()
    secuencias.append((secuencia.copy(),utilidad))
    return [utilidad]
  if etapa==1:
    valor=[-1000,None]
  else:
    valor=[1000,None]
  jugadas_posibles=juego.generar_jugadas_posibles()
  for jugada in jugadas_posibles:
    juego.jugar(jugada)
    secuencia.append(jugada)
    opcion=minimax(juego,etapa*-1,secuencia,secuencias)
    #maximizar
    if etapa==1:
      if valor[0]<opcion[0]:
        valor=[opcion[0],jugada]
    else:
    #minimizar
      if valor[0]>opcion[0]:
        valor=[opcion[0],jugada]
    juego.deshacer_jugada(jugada)
    secuencia.pop()
  return valor