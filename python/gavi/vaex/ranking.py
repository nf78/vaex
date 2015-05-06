from qt import *
import itertools
import gavifast
import gavi.dataset
import numpy as np
import gavi.vaex.plot_windows

import gavi.logging as logging
from pyjavaproperties import Properties
logger = logging.getLogger("vaex.ranking")


def unique_column_names(dataset):
	return list(set(dataset.column_names) | set(dataset.virtual_columns.keys()))
	
	
class ____RankingTableModel(QtCore.QAbstractTableModel):
	def __init__(self, dataset, dim=1, parent=None, *args): 
		QtCore.QAbstractTableModel.__init__(self, parent, *args) 
		self.dataset = dataset


		self.pairs = list(itertools.combinations(unique_column_names(self.dataset), dim))
		self.ranking = [None for pair in self.pairs]
		self.headers = ["subspace", "ranking", 'selected']
		self.indices = range(len(self.pairs))
	
	def rowCount(self, parent): 
		return len(self.pairs)

	def columnCount(self, parent): 
		return len(self.headers)

	def data(self, index, role): 
		if not index.isValid(): 
			return None
		elif role != QtCore.Qt.DisplayRole: 
			return None
		column = index.column()
		index = self.indices[index.row()] # use sorted index
		if column == 0:
			return "-vs".join(self.pairs[index])
		if column == 1:
			rank = self.ranking[index]
			return "" if rank is None else str(rank)
		if column == 2:
			rank = self.ranking[index]
			return False if random.random() < 0.5 else True

	def headerData(self, index, orientation, role):
		if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
			return self.headers[index]
		if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
			return str(index+1)
		return None

	def sort(self, Ncol, order):
		"""Sort table by given column number.
		"""
		self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
		if Ncol == 0:
			print "by name"
			# get indices, sorted by pair name
			sortlist = zip(self.pairs, range(len(self.pairs)))
			print sortlist
			sortlist.sort(key=operator.itemgetter(0))
			print sortlist
			self.indices = map(operator.itemgetter(1), sortlist)
			print self.indices
		if Ncol == 1:
			# get indices, sorted by ranking, or no sorting
			if None not in self.ranking:
				sortlist = zip(self.ranking, range(len(self.pairs)))
				sortlist.sort(key=operator.itemgetter(0))
				self.indices = map(operator.itemgetter(1), sortlist)
			else:
				self.indices = range(len(self.pairs))
			print self.indices
		if order == QtCore.Qt.DescendingOrder:
			self.indices.reverse()
		print self.indices
		self.emit(QtCore.SIGNAL("layoutChanged()"))

class SubspaceTable(QtGui.QTableWidget):
	def __init__(self, parent, mainPanel, dataset, pairs, dim, properties):
		self.headers = ['', 'space', 'ranking', 'plot']
		self.dim = dim
		if dim == 1:
			self.headers += ["min", "max"]
		self.properties = properties
		self.qualities = {}

		#print ", ".join([""+("-".join(pair))+"" for pair in pairs])
		self.dataset = dataset
		self.filter_terms = []
		self.mainPanel = mainPanel
		self.pairs = list(pairs) #list(itertools.combinations(self.dataset.column_names, dim))
		QtGui.QTableWidget.__init__(self, len(self.pairs), len(self.headers), parent)
		self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows);
		self.filter_mask = np.array([True for pair in pairs])
		self.selected_dict = {pair:self.properties.get(".".join(pair) + ".use", "True") == "True" for pair in pairs}
		#self.tableModel = RankingTableModel(self.dataset, dim, parent)
		#self.setModel(self.tableModel)
		#self.sortByColumn(0, QtCore.Qt.AscendingOrder)
		#self.setSortingEnabled(True)
		#self.pair_to_item = {}
		self.defaultFlags = QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsEditable
		#print self.properties._props
		if 1:
			#self.ranking = [None for pair in self.pairs]
			self.dim = dim
			self.setHorizontalHeaderLabels(self.headers)
			#self.setVerticalHeaderLabels(map(str, range(len(self.pairs))))
			self.fill_table()
			self.setSortingEnabled(True)
		self.queue_fill_table = gavi.vaex.plot_windows.Queue("fill table", 200, self.fill_table)

	def fill_table(self):
		# bug in qt? http://stackoverflow.com/questions/7960505/strange-qtablewidget-behavior-not-all-cells-populated-after-sorting-followed-b
		# fix: disable sorting, then enable again
		self.setSortingEnabled(False)
		self.checkboxes = []
		self.buttons = []
		pairs = [pair for pair, display in zip(self.pairs, self.filter_mask) if display]
		self.setRowCount(len(pairs))
		self.setVerticalHeaderLabels(map(str, range(len(pairs))))
		for i in range(len(pairs)):
			pair = pairs[i]
			text = " ".join(map(str, pair))
			item = QtGui.QTableWidgetItem(text)
			self.setItem(i, 1, item)
			item.setFlags(self.defaultFlags)
			#item = QtGui.QTableWidgetItem()
			#item.setData(QtCore.Qt.DisplayRole, QtCore.QVariant(True))
			#item.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
			checkbox = QtGui.QCheckBox(self)
			use_key = ".".join(map(str, pair)) + ".use"
			#if self.dim == 1 and use_key in self.properties._props:
			#	#print use_key, eval(self.properties[use_key])
			#	checkbox.setCheckState(QtCore.Qt.Checked if eval(self.properties[use_key]) else QtCore.Qt.Unchecked)
			#else:
			#	checkbox.setCheckState(QtCore.Qt.Checked)
			print "fill", pair, self.selected_dict[pair]
			checkbox.setCheckState(QtCore.Qt.Checked if self.selected_dict[pair] else QtCore.Qt.Unchecked)
			self.checkboxes.append(checkbox)
			self.setCellWidget(i, 0, checkbox)
			def stateChanged(state, pair=pair):
				self.selected_dict[pair] = state == QtCore.Qt.Checked
				print "set", pair, "to", self.selected_dict[pair]
			checkbox.stateChanged.connect(stateChanged)

			if self.dim == 1:
				button = QtGui.QPushButton("plot: " + text, self)
				def plot(_ignore=None, pair=pair):
					self.mainPanel.histogram(*pair)
				button.clicked.connect(plot)
				self.setCellWidget(i, 3, button)

				min_key = pair[0]+".min"
				max_key = pair[0]+".max"
				if 1:
					#print "test", min_key
					if min_key in self.properties._props:
						item = QtGui.QTableWidgetItem()#"%s"  % quality)
						value = self.properties[min_key]
						#print "it is in... and value =", value
						item.setText("%s"  % value)
						item.setData(QtCore.Qt.DisplayRole, float(value))
						item.setFlags(self.defaultFlags)
						self.setItem(i, 4, item)

					if max_key in self.properties._props:
						value = self.properties[max_key]
						item = QtGui.QTableWidgetItem()#"%s"  % quality)
						item.setText("%s"  % value)
						item.setData(QtCore.Qt.DisplayRole, float(value))
						item.setFlags(self.defaultFlags)
						self.setItem(i, 5, item)
			else:
				#print "quality", quality, qualities
				#row = self.pairs.index(pair)
				quality = self.qualities.get(pair)
				if quality is not None:
					item = QtGui.QTableWidgetItem()#"%s"  % quality)
					item.setText("%s"  % quality)
					item.setData(QtCore.Qt.DisplayRole, float(quality))
					item.setFlags(self.defaultFlags)
					self.setItem(i, 2, item)


			if self.dim == 2:
				button = QtGui.QPushButton("plot: " + text, self)
				def plot(_ignore=None, pair=pair):
					self.mainPanel.plotxy(*pair)
				button.clicked.connect(plot)
				self.setCellWidget(i, 3, button)
				self.buttons.append(button) # keep ref count
			#self.setItem(i, 1, item)
		self.setSortingEnabled(True)

	def getSelected(self):
		selection = [checkbox.checkState() == QtCore.Qt.Checked for checkbox in self.checkboxes]
		selected_pairs = [pair for pair, selected in zip(self.pairs, selection) if selected]
		return selected_pairs
		
	def setQualities(self, pairs, qualities):
		self.qualities = {}
		for quality, pair in zip(qualities, pairs):
			self.qualities[pair] = quality
			#item = self.pair_to_item[pair]
			#print "quality", quality, qualities
			row = self.pairs.index(pair)
			item = QtGui.QTableWidgetItem()#"%s"  % quality)
			item.setText("%s"  % quality)
			item.setData(QtCore.Qt.DisplayRole, float(quality))
			item.setFlags(self.defaultFlags)
			self.setItem(row, 2, item)

	def get_range(self, pair):
		index = self.pairs.index(pair)
		mi = self.item(index, 4)
		ma = self.item(index, 5)
		#print pair, mi, ma
		mi = None if mi is None else float(mi.data(QtCore.Qt.DisplayRole))
		ma = None if ma is None else float(ma.data(QtCore.Qt.DisplayRole))
		#print "->", pair, mi, ma
		return mi, ma

	def setRanges(self, pairs, ranges):
		for (mi, ma), pair in zip(ranges, pairs):
			#item = self.pair_to_item[pair]
			row = self.pairs.index(pair)
			item = QtGui.QTableWidgetItem()#"%s"  % quality)
			item.setText("%s"  % mi)
			item.setData(QtCore.Qt.DisplayRole, float(mi))
			item.setFlags(self.defaultFlags)
			self.setItem(row, 4, item)
			item = QtGui.QTableWidgetItem()#"%s"  % quality)
			item.setText("%s"  % ma)
			item.setData(QtCore.Qt.DisplayRole, float(ma))
			item.setFlags(self.defaultFlags)
			self.setItem(row, 5, item)
			

	def deselect(self, pair):
		index = self.pairs.index(pair)
		print "deselect", pair, index
		checkbox = self.checkboxes[index]
		checkbox.setCheckState(QtCore.Qt.Unchecked)
		
	def select(self, pair):
		index = self.pairs.index(pair)
		print "deselect", pair, index
		checkbox = self.checkboxes[index]
		checkbox.setCheckState(QtCore.Qt.Checked)

	def setPairs(self, pairs):
		#selection = [checkbox.checkState() == QtCore.Qt.Checked for checkbox in self.checkboxes]
		#non_selected_pairs = [pair for pair, selected in zip(self.pairs, selection) if not selected]

		self.pairs = list(pairs)
		for pair in self.pairs:
			if pair not in self.selected_dict:
				self.selected_dict[pair] = self.properties.get(".".join(pair) + ".use", "True")
		self.filter_mask = np.array([True for pair in pairs])
		self.fill_table()
		#self.checkboxes = []
		#self.setRowCount(len(self.pairs))
		#self.setVerticalHeaderLabels(map(str, range(len(self.pairs))))
		#for i in range(len(self.pairs)):
			#text = " ".join(map(str, self.pairs[i]))
			#print text
			#item = QtGui.QTableWidgetItem(text)
			#item.setFlags(self.defaultFlags)
			#self.setItem(i, 1, item)
			#checkbox = self.checkboxes[i] #QtGui.QCheckBox(self)
			#if not (self.pairs[i] in non_selected_pairs):
			#	checkbox.setCheckState(QtCore.Qt.Checked)
			#self.checkboxes.append(checkbox)
			#self.setCellWidget(i, 0, checkbox)
		#print self.checkboxes

	def set_filter_terms(self, filter_terms):
		def filter(pair):
			found = True
			for filter_term in filter_terms:
				found_term = False
				for expression in pair:
					found_term = found_term or filter_term.lower() in expression.lower()
				found = found and found_term
			return found

		self.filter_terms = filter_terms
		self.filter_mask = np.array([filter(pair) for pair in self.pairs])
		self.queue_fill_table()
		#self.fill_table()

import functools

def joinpairs(pairs1d, pairsNd):
	previous = []
	for pair1d in pairs1d:
		subspacename = pair1d[0] # tuple only has one element
		for pairNd in pairsNd:
			if subspacename not in pairNd:
				pair = pair1d + pairNd
				if sorted(pair) not in previous:
					previous.append(sorted(pair))
					#print previous
					yield pair

class RankDialog(QtGui.QDialog):
	def __init__(self, dataset, parent, mainPanel, **options):
		super(RankDialog, self).__init__(parent)
		self.dataset = dataset
		self.mainPanel = mainPanel
		self.range_map = {}


		#print "options", options
		self.properties = Properties()
		self.properties_path = os.path.splitext(self.dataset.path)[0] + ".properties"
		self.properties_path = options.get("file", self.properties_path)
		if os.path.exists(self.properties_path):
			self.properties.load(open(self.properties_path))
		else:
			pass
			#if not os.access(properties_path, os.W_OK):
			#	dialog_error(self, "File access", "Cannot write to %r, so cannot save options" % properties_path)



		
		self.tabs = QtGui.QTabWidget(self)
		
		self.tab1d = QtGui.QWidget(self.tabs)
		self.table1d = SubspaceTable(self.tab1d, mainPanel, self.dataset,  list(itertools.combinations(unique_column_names(self.dataset), 1)),  1, self.properties)
		
		self.subspaceTables = {}
		self.subspaceTabs = {}
		self.subspaceTables[1] = self.table1d
		self.subspaceTabs[1] = self.tab1d
		
		def onclick(dim=2):
			pairs1d = self.subspaceTables[1].getSelected()
			pairsprevd = self.subspaceTables[dim-1].getSelected()
			#print pairs1d
			#print pairsprevd
			newpairs = list(joinpairs(pairs1d, pairsprevd))
			print "newpairs", newpairs
			if dim not in self.subspaceTables:
				self.tabNd = QtGui.QWidget(self.tabs)
				self.tableNd = SubspaceTable(self.tabNd, self.mainPanel, self.dataset, newpairs, dim, self.properties)
				self.tabNdlayout = QtGui.QVBoxLayout(self)
				self.subspaceNd = QtGui.QPushButton("create %dd subspaces" % (dim+1), self.tab1d)
				self.rankNd = QtGui.QPushButton("rank subspaces")
				if dim == len(self.dataset.column_names):
					self.subspaceNd.setDisabled(True)
				self.tabNdlayout.addWidget(self.subspaceNd)
				self.tabNdlayout.addWidget(self.rankNd)
				self.subspaceNd.clicked.connect(functools.partial(onclick, dim=dim+1))
				self.rankNd.clicked.connect(functools.partial(self.rankSubspaces, table=self.tableNd))
				
				def func(index, name=""):
					print name, index.row(), index.column()
				self.tableNd.pressed.connect(functools.partial(func, name="pressed"))
				self.tableNd.entered.connect(functools.partial(func, name="entered"))
				self.tableNd.clicked.connect(functools.partial(func, name="clicked"))
				self.tableNd.activated.connect(functools.partial(func, name="activated"))
				def func(index, previous, name=""):
					print name, index.row(), index.column(), previous.row(), previous.column()
				self.selectionModel = self.tableNd.selectionModel()
				self.selectionModel.currentChanged.connect(functools.partial(func, name="currentChanged"))
				
				self.filter_Nd_line_edit = QtGui.QLineEdit(self)
				self.filter_Nd_line_edit.setPlaceholderText("Enter space seperated search terms")
				self.filter_Nd_line_edit.textEdited.connect(functools.partial(self.onFilter, table=self.tableNd))
				self.tabNdlayout.addWidget(self.filter_Nd_line_edit)

				self.tabNdlayout.addWidget(self.tableNd)
				#self.tab1dlayout.addWidget(self.rankButton)
				#self.setCentralWidget(self.splitter)
				self.tabNd.setLayout(self.tabNdlayout)
				self.subspaceTables[dim] = self.tableNd
				self.subspaceTabs[dim] = self.tabNd
				
				self.tabs.addTab(self.tabNd, "%dd" % dim)
				self.tabs.setCurrentWidget(self.tabNd)
			else:
				self.subspaceTables[dim].setPairs(newpairs)
				self.tabs.setCurrentWidget(self.subspaceTabs[dim])

		self.subspace2d = QtGui.QPushButton("create 2d subspaces", self.tab1d)
		self.subspace2d.clicked.connect(functools.partial(onclick, dim=2))

		self.get_ranges_menu = QtGui.QMenu()
		self.button_get_ranges = QtGui.QToolButton()
		self.button_get_ranges.setText("calculate min/max")
		self.button_get_ranges.setPopupMode(QtGui.QToolButton.InstantPopup)
		#self.button_get_ranges.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.get_ranges_menu = QtGui.QMenu()
		self.button_get_ranges.setMenu(self.get_ranges_menu)

		self.action_ranges_minmax = QtGui.QAction("min/max", self)
		self.get_ranges_menu.addAction(self.action_ranges_minmax)

		#self.button_get_ranges = QtGui.QToolButton(self.tab1d)
		#self.button_get_ranges.setText("calculate min/max")
		#self.button_get_ranges.setM
		self.action_ranges_minmax.triggered.connect(self.onCalculateMinMax)
		
		self.button_store = QtGui.QToolButton(self.tab1d)
		self.button_store.setText("store")
		self.button_store.clicked.connect(self.onStore)


		self.actions_menu = QtGui.QMenu()
		self.button_actions = QtGui.QToolButton()
		self.button_actions.setText("Extra")
		self.button_actions.setPopupMode(QtGui.QToolButton.InstantPopup)
		self.button_actions.setMenu(self.actions_menu)

		self.action_select_all = QtGui.QAction("Select all", self)
		self.action_select_none = QtGui.QAction("Select none", self)
		self.action_remove_empty = QtGui.QAction("Remove empty columns", self)
		self.action_select_all.triggered.connect(self.onSelectAll)
		self.action_select_none.triggered.connect(self.onSelectNone)
		self.action_remove_empty.triggered.connect(self.onRemoveEmpty)
		self.actions_menu.addAction(self.action_select_all)
		self.actions_menu.addAction(self.action_select_none)
		self.actions_menu.addAction(self.action_remove_empty)

		self.tab1dlayout = QtGui.QVBoxLayout(self)
		self.tab1d_button_layout = QtGui.QHBoxLayout(self)
		self.tab1dlayout.addLayout(self.tab1d_button_layout)
		self.tab1d_button_layout.addWidget(self.subspace2d)
		self.tab1d_button_layout.addWidget(self.button_get_ranges)
		self.tab1d_button_layout.addWidget(self.button_store)
		self.tab1d_button_layout.addWidget(self.button_actions)


		self.filter_line_edit = QtGui.QLineEdit(self)
		self.filter_line_edit.setPlaceholderText("Enter space seperated search terms")
		self.filter_line_edit.textEdited.connect(functools.partial(self.onFilter, table=self.table1d))

		self.tab1dlayout.addWidget(self.filter_line_edit)
		self.tab1dlayout.addWidget(self.table1d)
		#self.tab1dlayout.addWidget(self.rankButton)
		#self.setCentralWidget(self.splitter)
		self.tab1d.setLayout(self.tab1dlayout)
		
		self.tabs.addTab(self.tab1d, "1d")
		
		self.resize(700,500)
		
		if 0:
			for name in self.dataset.column_names:
				item = QtGui.QListWidgetItem(self.list1d)
				item.setText(name)
				item.setCheckState(False)
				#self.list1d.


		self.boxlayout = QtGui.QVBoxLayout(self)
		self.radio_button_all = QtGui.QRadioButton("Use complete dataset", self)
		self.radio_button_selection = QtGui.QRadioButton("Use selection", self)
		self.radio_button_all.setChecked(True)
		self.boxlayout.addWidget(self.radio_button_all)
		self.boxlayout.addWidget(self.radio_button_selection)
		self.boxlayout.addWidget(self.tabs)
		#self.boxlayout.addWidget(self.rankButton)
		#self.setCentralWidget(self.splitter)
		self.setLayout(self.boxlayout)


		if "2" in options.get("open", ""):
			onclick(dim=2)
		if "3" in options.get("open", ""):
			onclick(dim=3)
		if "4" in options.get("open", ""):
			onclick(dim=4)


	def onFilter(self, text, table):
		table.set_filter_terms(text.split())

	def onStore(self):
		selected_pairs = self.table1d.getSelected()
		#error = False
		for pair in self.table1d.pairs:
			key = str(pair[0])
			#print repr(key+".use"), repr(pair in selected_pairs)
			self.properties[key+".use"] = repr(pair in selected_pairs)
			if key in self.range_map:
				mi, ma = self.range_map[key]
				self.properties[key+".min"] = repr(mi)
				self.properties[key+".max"] = repr(ma)
			else:
				print "min/max not present", key
		print "save to", self.properties_path
		self.properties.store(open(self.properties_path, "w"))
		dialog_info(self, "Stored", "Stored configuration to: %r" % self.properties_path)

	def fill_range_map(self):
		pairs = self.table1d.getSelected()
		for pair in pairs:
			mi, ma = self.table1d.get_range(pair)
			if mi is not None and ma is not None:
				self.range_map[pair[0]] = mi, ma

	def onSelectAll(self):
		pairs = self.table1d.pairs
		for pair in pairs:
			self.table1d.select(pair)

	def onSelectNone(self):
		pairs = self.table1d.pairs
		for pair in pairs:
			self.table1d.deselect(pair)

	def onRemoveEmpty(self):
		self.fill_range_map()
		pairs = self.table1d.getSelected()
		error = False
		for pair in pairs:
			print pair
			if pair[0] in self.range_map:
				min, max = self.range_map[pair[0]]
				if min == max:
					print pair, "empty", min, max
					self.table1d.deselect(pair)
			else:
				if not error: # only give a warning once
					dialog_error(self, "Min/max missing", "Min max missing for %s" % pair)
					error = True


	def onCalculateMinMax(self):
		pairs = self.table1d.getSelected()
		logger.debug("estimate min/max for %r" % pairs)
		jobsManager = gavi.dataset.JobsManager()
		expressions = [pair[0] for pair in pairs]
		assert len(pairs[0]) == 1
		self.range_map = {}
		dialog = QtGui.QProgressDialog("Calculating min/max", "Abort", 0, 1000, self)
		dialog.show()
		def feedback(percentage):
			dialog.setValue(int(percentage*10))
			QtCore.QCoreApplication.instance().processEvents()
			if dialog.wasCanceled():
				return True
		try:
			ranges = jobsManager.find_min_max(self.dataset, expressions, use_mask=self.radio_button_selection.isChecked(), feedback=feedback)
			for range_, expression in zip(ranges, expressions):
				logger.debug("range for {expression} is {range_}".format(**locals()))
				self.range_map[expression] = range_
			self.table1d.setRanges(pairs, ranges)
		except:
			logger.exception("Error in min/max or cancelled")
		dialog.hide()

	def rankSubspaces(self, table):
		self.fill_range_map()
		print table
		qualities = []
		pairs = table.getSelected()
		error = False
		print self.range_map
		for pair in pairs:
			for expression in pair:
				if expression not in self.range_map:
					error = True
					print "missing", expression
		if error:
			dialog_error(self, "Missing min/max", "Please calculate the minimum and maximum for the dimensions")
			return

		if 0:
			for pair in pairs:
				dim = len(pair)
				#if dim == 2:
				columns = [self.dataset.columns[name] for name in pair]
				print pair
				information = gavi.kld.kld_shuffled(columns, mask=mask)
				qualities.append(information)
				#print pair
		dialog = QtGui.QProgressDialog("Calculating KL divergence", "Abort", 0, 1000, self)
		dialog.show()
		def feedback(percentage):
			dialog.setValue(int(percentage*10))
			QtCore.QCoreApplication.instance().processEvents()
			if dialog.wasCanceled():
				return True
		qualities = gavi.kld.kld_shuffled_grouped(self.dataset, self.range_map, pairs, feedback=feedback, use_mask=self.radio_button_selection.isChecked())
		if qualities is not None:
			print qualities
			table.setQualities(pairs, qualities)
		
