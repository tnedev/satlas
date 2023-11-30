from satlas_interface import Satlas

sat = Satlas()
sat.load_image('./images/test_image.png')

# Run results on Polygon task
sat.evaluate(task=Satlas.Task.POLYGON, add_legend=True)

# Run results on Point task
sat.evaluate(task=Satlas.Task.POINT, add_legend=True)

# Run results on PARK_SPORT (classification) task
sat.evaluate(task=Satlas.Task.PARK_SPORT)
