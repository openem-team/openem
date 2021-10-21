const svgNamespace = "http://www.w3.org/2000/svg";

class TatorElement extends HTMLElement {
  constructor() {
    super();
    this._shadow = this.attachShadow({mode: "open"});

    const css = document.createElement("link");
    css.setAttribute("rel", "stylesheet");
    css.setAttribute("href", "tator.min.css");
    this._shadow.appendChild(css);

    css.addEventListener("load", evt => {
      this.style.visibility = "visible";
    });
  }

  connectedCallback() {
    this.style.visibility = "hidden";
  }
}
class TextInput extends TatorElement {
  constructor() {
    super();

    this.label = document.createElement("label");
    this.label.setAttribute("class", "d-flex flex-justify-between flex-items-center py-1");
    this._shadow.appendChild(this.label);

    this._name = document.createTextNode("");
    this.label.appendChild(this._name);

    this._input = document.createElement("input");
    this._input.setAttribute("class", "form-control input-sm col-8");
    this._input.setAttribute("type", "text");
    this.label.appendChild(this._input);

    this._input.addEventListener("change", () => {
      if (this.getValue() === null) {
        this._input.classList.add("has-border");
        this._input.classList.add("is-invalid");
      } else {
        this._input.classList.remove("has-border");
        this._input.classList.remove("is-invalid");
      }
      this.dispatchEvent(new Event("change"));
    });

    this.getValue = this._validateString;

    this._input.addEventListener("focus", () => {
      document.body.classList.add("shortcuts-disabled");
    });

    this._input.addEventListener("blur", () => {
      document.body.classList.remove("shortcuts-disabled");
    });

  }

  static get observedAttributes() {
    return ["name", "type"];
  }

  attributeChangedCallback(name, oldValue, newValue) {
    switch (name) {
      case "name":
        this._name.nodeValue = newValue;
        break;
      case "type":
        switch (newValue) {
          case "int":
            this._input.setAttribute("placeholder", "Enter an integer");
            this.getValue = this._validateInt;
            break;
          case "float":
            this._input.setAttribute("placeholder", "Enter a number");
            this.getValue = this._validateFloat;
            break;
          case "string":
            this.getValue = this._validateString;
            break;
          case "datetime":
            this._input.setAttribute("placeholder", "e.g. 2020-06-30");
            this.getValue = this._validateDateTime;
            break;
          case "geopos":
            this._input.setAttribute("placeholder", "e.g. 21.305,-157.858");
            this.getValue = this._validateGeopos;
            break;
          case "password":
            this.getValue = this._validatePassword;
            this._input.setAttribute("type", "password");
            break;
          case "email":
            this.getValue = this._validateEmail;
            this._input.setAttribute("type", "email");
            break;
          default:
            this._input.setAttribute("type", newValue);
            break
        }
        break;
    }
  }

  set permission(val) {
    if (hasPermission(val, "Can Edit")) {
      this._input.removeAttribute("readonly");
      this._input.classList.remove("disabled");
    } else {
      this._input.setAttribute("readonly", "");
      this._input.classList.add("disabled");
    }
  }

  set default(val) {
    this._default = val;
  }

  /**
   * @param {boolean} val
   */
  set disabled(val) {
    this._input.disabled = val;
  }

  changed(){
    return this.getValue() !== this._default;
  }

  reset() {
    // Go back to default value
    if (typeof this._default !== "undefined") {
      this.setValue(this._default);
    } else {
      this.setValue("");
    }
  }

  _validateInt() {
    let val = parseInt(this._input.value);
    if (isNaN(val)) {
      val = null;
    }
    return val;
  }

  _validateFloat() {
    let val = parseFloat(this._input.value);
    if (isNaN(val)) {
      val = null;
    }
    return val;
  }

  _validateString() {
    return this._input.value;
  }

  _validateDateTime() {
    let val = new Date(this._input.value);
    if (isNaN(val.getTime())) {
      val = null;
    } else {
      val = val.toISOString();
    }
    return val;
  }

  _validateGeopos() {
    const val = this._input.value.split(",");
    let ret = null;
    if (val.length == 2) {
      const lat = parseFloat(val[0]);
      const lon = parseFloat(val[1]);
      if (!isNaN(lat) && !isNaN(lon)) {
        const latOk = (lat < 90.0) && (lat > -90.0);
        const lonOk = (lon < 180.0) && (lon > -180.0);
        if (latOk && lonOk) {
          ret = [lat, lon];
        }
      }
    }
    return ret;
  }

  _validatePassword() {
    return this._input.value;
  }

  _validateEmail() {
    return this._input.value;
  }

  setValue(val) {
    this._input.value = val;
  }

  set autocomplete(config)
  {
    TatorAutoComplete.enable(this._input, config);
  }
}

customElements.define("text-input", TextInput);
class EnumInput extends TatorElement {
  constructor() {
    super();

    this.label = document.createElement("label");
    this.label.setAttribute("class", "d-flex flex-justify-between flex-items-center py-1");
    this.label.style.position = "relative";
    this._shadow.appendChild(this.label);

    this._name = document.createTextNode("");
    this.label.appendChild(this._name);

    const span = document.createElement("span");
    span.setAttribute("class", "sr-only");
    this.label.appendChild(span);

    this._select = document.createElement("select");
    this._select.setAttribute("class", "form-select has-border select-sm col-8");
    this.label.appendChild(this._select);

    // Add unselectable option for null values.
    this._null = document.createElement("option");
    this._null.setAttribute("value", "");
    this._null.setAttribute("disabled", "");
    this._null.setAttribute("hidden", "");
    this._null.textContent = "null";
    this._select.appendChild(this._null);

    // Add unselectable option for undefined values.
    this._undefined = document.createElement("option");
    this._undefined.setAttribute("value", "");
    this._undefined.setAttribute("disabled", "");
    this._undefined.setAttribute("hidden", "");
    this._undefined.textContent = "undefined";
    this._select.appendChild(this._undefined);

    const svg = document.createElementNS(svgNamespace, "svg");
    svg.setAttribute("class", "text-gray");
    svg.setAttribute("id", "icon-chevron-down");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("height", "1em");
    svg.setAttribute("width", "1em");
    this.label.appendChild(svg);

    const path = document.createElementNS(svgNamespace, "path");
    path.setAttribute("d", "M5.293 9.707l6 6c0.391 0.391 1.024 0.391 1.414 0l6-6c0.391-0.391 0.391-1.024 0-1.414s-1.024-0.391-1.414 0l-5.293 5.293-5.293-5.293c-0.391-0.391-1.024-0.391-1.414 0s-0.391 1.024 0 1.414z");
    svg.appendChild(path);

    this._select.addEventListener("change", () => {
      this.dispatchEvent(new Event("change"));
    });

  }

  static get observedAttributes() {
    return ["name"];
  }

  attributeChangedCallback(name, oldValue, newValue) {
    switch (name) {
      case "name":
        this._name.nodeValue = newValue;
        break;
    }
  }

  set permission(val) {
    if (hasPermission(val, "Can Edit")) {
      this._select.removeAttribute("disabled");
    } else {
      this._select.setAttribute("disabled", "");
    }
  }

  set choices(val) {
    let selectedDefault = null;
    // Add attribute type choices.
    for (const choice of val) {
      const option = document.createElement("option");
      option.setAttribute("value", choice.value);
      if ('label' in choice) {
        option.textContent = choice.label;
      } else {
        option.textContent = choice.value;
      }
      if (choice.selected) {
        selectedDefault = choice.value;
      }
      this._select.appendChild(option);
    }
    if (selectedDefault !== null) {
      this.setValue(selectedDefault)
    }
  }

  set default(val) {
    this._default = val;
  }

  // checks if the current value equals the default
  changed(){
    return this.getValue() !== this._default;
  }

  reset() {
    // Go back to default value
    if (typeof this._default !== "undefined") {
      this.setValue(this._default);
    } else {
      this._undefined.setAttribute("selected", "");
    }
  }

  getValue() {
    if (this._select.options.length !== 0) {
      const selected = this._select.selectedIndex;
      if(typeof this._select.options[selected] !== "undefined"){
        return this._select.options[selected].value;
      }
    }

    return null;
  }

  setValue(val) {
    let idx = 0;
    if (typeof val === "undefined") {
      this._undefined.setAttribute("selected", "");
    } else if (val === null) {
      this._null.setAttribute("selected", "");
    } else {
      for (const option of this._select.options) {
        if (option.value == val) {
          this._select.selectedIndex = idx;
          break;
        }
        idx++;
      }
    }
  }

  getChoices() {
    var choiceList = [];
    for (const option of this._select.options) {
      choiceList.push(option.value);
    }
    return choiceList;
  }


  /**
   * Clears the options. Useful for resetting the menu options.
   */
  clear() {
    while (this._select.options.length > 0) {
      this._select.options.remove(0);
    }
  }
}

customElements.define("enum-input", EnumInput);

class FancyTimeline extends TatorElement {
  constructor() {
    super();

    this._mainTimelineDiv = document.createElement("div");
    this._mainTimelineDiv.setAttribute("class", "py-2");
    this._mainTimelineDiv.id = "main-timeline";
    this._shadow.appendChild(this._mainTimelineDiv);

    this._focusTimelineDiv = document.createElement("div");
    this._focusTimelineDiv.setAttribute("class", "");
    this._focusTimelineDiv.id = "focus-timeline";
    this._shadow.appendChild(this._focusTimelineDiv);

    this._mainSvg = d3.select(this._shadow).select("#main-timeline")
      .append("svg")
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("font", "12px sans-serif")
      .style("color", "#a2afcd");

    this._focusSvg = d3.select(this._shadow).select("#focus-timeline")
      .append("svg")
      .attr("preserveAspectRatio", "xMidYMid meet")
      .style("font", "12px sans-serif")
      .style("color", "#a2afcd");

    this._focusLine = this._focusSvg.append("g").attr("display", "none");

    // Initially hide the focus timeline. Some external UI element will control
    // whether or not to display this.
    this._focusTimelineDiv.style.display = "none";
    
    // Redraw whenever there's a resize
    this._maxFrame = 1234;
    this._numericalData = [];
    this._stateData = [];
    window.addEventListener("resize", this._updateSvgData());
  }

  /**
   * #TODO
   */
  _getMaxFrame() {
    return this._maxFrame;
  }

  /**
   * Used in making unique identifiers for the various d3 graphing elements
   * @returns {object} id, href properties
   */
   _d3UID() {
    var id = uuidv1();
    var href = new URL(`#${id}`, location) + "";
    return {id: id, href: href};
  }

  /**
   * Called whenever there's new data to be displayed on the timelines
   */
   _updateSvgData() {

    var that = this;
    var maxFrame = this._getMaxFrame();
    if (isNaN(maxFrame)) {
      return;
    }

    this._mainLineHeight = 60;
    if (this._numericalData.length == 0) {
      this._mainLineHeight = 0;
    }
    this._mainStepPad = 2;
    this._mainStep = 5; // vertical height of each entry in the series / band
    this._mainMargin = ({top: 20, right: 3, bottom: 3, left: 3});
    this._mainHeight =
    this._mainLineHeight +
      this._stateData.length * (this._mainStep + this._mainStepPad) +
      this._mainMargin.top + this._mainMargin.bottom;
    this._mainWidth = this._mainTimelineDiv.offsetWidth;

    if (this._mainWidth <= 0) { return; }
    this._mainSvg.attr("viewBox",`0 0 ${this._mainWidth} ${this._mainHeight}`);

    // Define the axes
    this._mainX = d3.scaleLinear()
      .domain([0, maxFrame])
      .range([0, this._mainWidth])

    var mainY = d3.scaleLinear()
      .domain([0, 1.0])
      .range([0, -this._mainStep]);

    // #TODO This is clunky and has no smooth transition, but it works for our application
    //       Potentially worth revisiting in the future and updating the dataset directly
    //       using the traditional d3 enter/update/exit paradigm.
    this._mainSvg.selectAll('*').remove();

    // Frame number x-axis ticks
    if (this._numericalData.length == 0 && this._stateData.length == 0) {
      var xAxis = g => g
        .call(d3.axisBottom(this._mainX).ticks().tickSizeOuter(0).tickFormat(d3.format("d")))
        .call(g => g.selectAll(".tick").filter(d => this._mainX(d) < this._mainMargin.left || this._mainX(d) >= this._mainWidth - this._mainMargin.right).remove())
        .call(g => g.select(".domain").remove());
    }
    else {
      var xAxis = g => g
      .attr("transform", `translate(0,${this._mainMargin.top})`)
      .call(d3.axisTop(this._mainX).ticks().tickSizeOuter(0).tickFormat(d3.format("d")))
      .call(g => g.selectAll(".tick").filter(d => this._mainX(d) < this._mainMargin.left || this._mainX(d) >= this._mainWidth - this._mainMargin.right).remove())
      .call(g => g.select(".domain").remove());
    }

    // States are represented as area graphs
    var area = d3.area()
      .curve(d3.curveStepAfter)
      .x(d => this._mainX(d.frame))
      .y0(0)
      .y1(d => mainY(d.value));

    var mainStateDataset = this._stateData.map(d => Object.assign({
      clipId: this._d3UID(),
      pathId: this._d3UID(),
    }, d));

    const gState = this._mainSvg.append("g")
      .selectAll("g")
      .data(mainStateDataset)
      .join("g")
        .attr("transform", (d, i) => `translate(0,${i * (this._mainStep + this._mainStepPad) + this._mainMargin.top})`);

    gState.append("clipPath")
      .attr("id", d => d.clipId.id)
      .append("rect")
        .attr("width", this._mainWidth)
        .attr("height", this._mainStep);

    gState.append("defs").append("path")
      .attr("id", d => d.pathId.id)
      .attr("d", d => area(d.graphData));

    gState.append("rect")
      .attr("clip-path", d => d.clipId)
      .attr("fill", "#262e3d")
      .attr("width", this._mainWidth)
      .attr("height", this._mainStep);

    gState.append("g")
        .attr("clip-path", d => d.clipId)
      .selectAll("use")
      .data(d => new Array(1).fill(d))
      .join("use")
        .attr("fill", (d, i) => d.color)
        .attr("transform", (d, i) => `translate(0,${(i + 1) * this._mainStep})`)
        .attr("xlink:href", d => d.pathId.href);

    // Numerical data are represented as line graphs
    var mainLineDataset = this._numericalData.map(d => Object.assign({
      clipId: this._d3UID(),
      pathId: this._d3UID(),
      name: d.name
    }, d));

    var mainLineY = d3.scaleLinear()
      .domain([-0.1, 1.1])
      .range([0, -this._mainLineHeight]);

    var mainLine = d3.line()
      .curve(d3.curveStepAfter)
      .x(d => this._mainX(d.frame))
      .y(d => mainLineY(d.value));

    const startOfMainLineGraph = (this._stateData.length) * (this._mainStep + this._mainStepPad) + this._mainMargin.top;

    if (mainLineDataset.length > 0) {
      this._mainSvg.append("rect")
        .attr("transform", `translate(0,${startOfMainLineGraph})`)
        .attr("fill", "#262e3d")
        .attr("width", this._mainWidth)
        .attr("height", this._mainLineHeight);
    }

    this._mainLineG = this._mainSvg.append("g")
      .selectAll("g")
      .data(mainLineDataset)
      .join("g")
        .attr("transform", `translate(0,${startOfMainLineGraph})`);

    this._mainLineText = this._mainLineG.append("text")
      .attr("x", 4)
      .attr("y", this._mainLineHeight / 2)
      .attr("dy", "0.35em")
      .attr("fill", "#fafafa")
      .attr("opacity", "0.0");

    this._mainLineG.append("clipPath")
      .attr("id", d => d.clipId.id)
      .append("rect")
        .attr("width", this._mainWidth)
        .attr("height", this._mainLineHeight);

    this._mainLineG.append("defs").append("path")
      .attr("id", d => d.pathId.id)
      .attr("d", d => mainLine(d.graphData));

    this._mainLineG.append("g")
      .attr("clip-path", d => d.clipId)
      .selectAll("use")
      .data(d => new Array(1).fill(d))
      .join("use")
        .attr("opacity","0.7")
        .attr("stroke", d => "#797991")
        .attr("stroke-width", d => 1.0)
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("fill", "none")
        .attr("transform", `translate(0,${this._mainLineHeight})`)
        .attr("xlink:href", d => d.pathId.href)
        .style("stroke-dasharray", ("1, 2"));

    this._mainFrameLine = this._mainSvg.append("line")
      .attr("stroke", "#fafafa")
      .attr("stroke-width", 1)
      .attr("opacity", "0");

    this._mainSvg
      .on("mousemove", function(event) {
        event.preventDefault();

        // Remember the y-axis is 0 to -1
        const pointer = d3.pointer(event, that);
        const pointerFrame = that._mainX.invert(pointer[0]);
        const pointerValue = mainLineY.invert(pointer[1] - startOfMainLineGraph) + 1.0;

        var selectedData;
        var currentDistance;
        var selectedDistance = Infinity;
        for (let datasetIdx = 0; datasetIdx < mainLineDataset.length; datasetIdx++) {

          let d = mainLineDataset[datasetIdx];

          for (let idx = 0; idx < d.graphData.length; idx++) {
            if (d.graphData[idx].frame > pointerFrame) {
              if (idx > 0) {

                currentDistance = Math.abs(pointerValue - d.graphData[idx - 1].value);
                if (currentDistance < selectedDistance) {
                  selectedData = d;
                  selectedDistance = currentDistance
                }

                break;

              }
            }
          }
        }

        if (typeof selectedData != "undefined") {
          that._highlightMainLine(selectedData.name);
        }
      })
    .on("mouseleave", function() {
      that._unhighlightMainLines();
    })

    // Add the x-axis
    this._mainSvg.append("g")
      .style("font-size", "12px")
      .call(xAxis);

    // Setup the brush to focus/zoom on the main timeline
    this._mainBrush = d3.brushX()
      .extent([[this._mainMargin.left, 0.5], [this._mainWidth - this._mainMargin.right, this._mainHeight - this._mainMargin.bottom + 0.5]])
      .on("end", this._mainBrushEnded.bind(this))
      .on("brush", this._mainBrushed.bind(this));

    // The brush will default to nothing being selected
    this._mainBrushG = this._mainSvg.append("g")
      .call(this._mainBrush);

    this._mainBrushG
      .call(this._mainBrush.move, null);
  }

  /**
   * Callback for "brush" with d3.brushX
   * This recreates the focusSVG/timeline with the dataset the brush on the mainSVG covers
   * @param {array} selection Mouse pointer event
   */
   _mainBrushed ({selection}) {

    if (!selection) {
      return;
    }

    // Selection is an array of startX and stopX
    // Use this to update the x-axis of the focus panel
    const focusStep = 25; // vertical height of each entry in the series / band
    const focusStepPad = 4;
    const focusMargin = ({top: 20, right: 5, bottom: 3, left: 5});
    const focusHeight =
      this._numericalData.length * (focusStep + focusStepPad) +
      this._stateData.length * (focusStep + focusStepPad) +
      focusMargin.top + focusMargin.bottom;
    const focusWidth = this._mainWidth;
    this._focusSvg.attr("viewBox",`0 0 ${focusWidth} ${focusHeight}`);

    // Define the axes
    var minFrame = this._mainX.invert(selection[0]);
    var focusX = d3.scaleLinear()
      .domain([minFrame, this._mainX.invert(selection[1])])
      .range([0, focusWidth]);

    var focusY = d3.scaleLinear()
      .domain([0, 1.0])
      .range([0, -focusStep]);

    this.dispatchEvent(new CustomEvent("zoomedTimeline", {
      composed: true,
      detail: {
        minFrame: Math.round(minFrame),
        maxFrame: Math.round(this._mainX.invert(selection[1]))
      }
    }));

    // #TODO This is clunky and has no smooth transition, but it works for our application
    //       Potentially worth revisiting in the future and updating the dataset directly
    //       using the traditional d3 enter/update/exit paradigm.
    this._focusSvg.selectAll('*').remove();

    // X-axis that will be displayed to visualize the frame numbers
    var focusXAxis = g => g
      .attr("transform", `translate(0,${focusMargin.top})`)
      .call(d3.axisTop(focusX).ticks().tickSizeOuter(0).tickFormat(d3.format("d")))
      .call(g => g.selectAll(".tick").filter(d => focusX(d) < focusMargin.left || focusX(d) >= focusWidth - focusMargin.right).remove())
      .call(g => g.select(".domain").remove());

    // States are represented as area graphs
    var focusArea = d3.area()
      .curve(d3.curveStepAfter)
      .x(d => focusX(d.frame))
      .y0(0)
      .y1(d => focusY(d.value));

    var focusLine = d3.line()
      .curve(d3.curveStepAfter)
      .x(d => focusX(d.frame))
      .y(d => focusY(d.value));

    var focusStateDataset = this._stateData.map(d => Object.assign({
        clipId: this._d3UID(),
        pathId: this._d3UID(),
        textId: this._d3UID()
      }, d));

    const focusG = this._focusSvg.append("g")
      .selectAll("g")
      .data(focusStateDataset)
      .join("g")
        .attr("transform", (d, i) => `translate(0,${i * (focusStep + focusStepPad) + focusMargin.top})`);

    focusG.append("clipPath")
      .attr("id", d => d.clipId.id)
      .append("rect")
        .attr("width", focusWidth)
        .attr("height", focusStep);

    if (minFrame > -1) {
      focusG.append("defs").append("path")
        .attr("id", d => d.pathId.id)
        .attr("d", d => focusArea(d.graphData));
    }

    focusG.append("rect")
      .attr("clip-path", d => d.clipId)
      .attr("fill", "#262e3d")
      .attr("width", focusWidth)
      .attr("height", focusStep);

    focusG.append("g")
        .attr("clip-path", d => d.clipId)
      .selectAll("use")
      .data(d => new Array(1).fill(d))
      .join("use")
        .attr("fill", (d, i) => d.color)
        .attr("transform", (d, i) => `translate(0,${(i + 1) * focusStep})`)
        .attr("xlink:href", d => d.pathId.href);

    // Unlike the main SVG, this SVG will display the corresponding attribute name
    // and the value when the user hovers over the SVG
    focusG.append("text")
        .attr("x", 4)
        .attr("y", focusStep / 2)
        .attr("dy", "0.5em")
        .attr("fill", "#fafafa")
        .style("font-size", "12px")
        .text(d => d.name);

    const focusStateValues = focusG.append("text")
        .attr("class", "focusStateValues")
        .attr("x", focusWidth * 0.4)
        .attr("y", focusStep / 2)
        .attr("dy", "0.5em")
        .style("font-size", "12px")
        .attr("fill", "#fafafa");

    // States are represented as line graphs
    var focusLineDataset = this._numericalData.map(d => Object.assign({
        clipId: this._d3UID(),
        pathId: this._d3UID(),
        textId: this._d3UID()
      }, d));

    const focusGLine = this._focusSvg.append("g")
      .selectAll("g")
      .data(focusLineDataset)
      .join("g")
        .attr("transform", (d, i) => `translate(0,${(i + this._stateData.length) * (focusStep + focusStepPad) + focusMargin.top})`);

    focusGLine.append("clipPath")
      .attr("id", d => d.clipId.id)
      .append("rect")
        .attr("width", focusWidth)
        .attr("height", focusStep);

    if (minFrame > -1){
      focusGLine.append("defs").append("path")
      .attr("id", d => d.pathId.id)
      .attr("d", d => focusLine(d.graphData));
    }

    focusGLine.append("rect")
      .attr("clip-path", d => d.clipId)
      .attr("fill", "#262e3d")
      .attr("width", focusWidth)
      .attr("height", focusStep);

    focusGLine.append("g")
        .attr("clip-path", d => d.clipId)
      .selectAll("use")
      .data(d => new Array(1).fill(d))
      .join("use")
        .attr("pointer-events", "none")
        .attr("stroke", (d, i) => "#797991")
        .attr("fill", (d, i) => "none")
        .attr("transform", (d, i) => `translate(0,${(i + 1) * focusStep})`)
        .attr("xlink:href", d => d.pathId.href)

    focusGLine.selectAll("rect")
      .on("mouseover", function(event, d) {
        that._highlightMainLine(d.name);
      })
      .on("mouseout", function(event, d) {
        that._unhighlightMainLines();
      });

    // Unlike the main SVG, this SVG will display the corresponding attribute name
    // and the value when the user hovers over the SVG
    focusGLine.append("text")
        .style("font-size", "12px")
        .attr("pointer-events", "none")
        .attr("x", 4)
        .attr("y", focusStep / 2)
        .attr("dy", "0.5em")
        .attr("fill", "#fafafa")
        .text(d => d.name);

    const focusLineValues = focusGLine.append("text")
        .style("font-size", "12px")
        .attr("class", "focusLineValues")
        .attr("pointer-events", "none")
        .attr("x", focusWidth * 0.4)
        .attr("y", focusStep / 2)
        .attr("dy", "0.5em")
        .attr("fill", "#fafafa");

    // Apply the x-axis ticks at the end, after the other graphics have been filled in
    var displayXAxis = selection[0] >= 0;
    if (displayXAxis) {
      var focusXAxisG = this._focusSvg.append("g")
        .style("font-size", "12px")
        .call(focusXAxis);

      var focusFrameTextBackground = focusXAxisG.append("rect")
        .attr("width", focusWidth)
        .attr("height", focusStep);

      var focusFrameText = focusXAxisG.append("text")
        .style("font-size", "12px")
        .attr("x", focusWidth * 0.4)
        .attr("y", -focusStep / 2)
        .attr("dy", "0.35em")
        .attr("fill", "#fafafa");
    }

    // Create the vertical line hover
    const mouseLine = this._focusSvg.append("line")
      .attr("pointer-events", "none")
      .attr("stroke", "#fafafa")
      .attr("stroke-width", 1)
      .attr("opacity", "0");

    var that = this;
    this._focusSvg.on("click", function(event, d) {

      const selectedFrame = focusX.invert(d3.pointer(event)[0]);
      const maxFrame = that._getMaxFrame();

      if (selectedFrame >= 0 && selectedFrame <= maxFrame) {
        that.dispatchEvent(new CustomEvent("select", {
          detail: {
            frame: selectedFrame
          }
        }));
      }
    });
    this._focusSvg.on("mouseover", function() {
        mouseLine.attr("opacity", "0.5");
        that._mainFrameLine.attr("opacity", "0.5");
    });
    this._focusSvg.on("mouseout", function() {
        mouseLine.attr("opacity", "0");
        that._mainFrameLine.attr("opacity", "0");
    });
    this._focusSvg.on("mousemove", function(event, d) {

        var currentFrame = parseInt(focusX.invert(d3.pointer(event)[0]));

        mouseLine
          .attr("opacity", "0.5")
          .attr("x1", d3.pointer(event)[0])
          .attr("x2", d3.pointer(event)[0])
          .attr("y1", -focusStep - focusMargin.bottom)
          .attr("y2", focusHeight);

        that._mainFrameLine
          .attr("opacity", "0.5")
          .attr("x1", that._mainX(currentFrame))
          .attr("x2", that._mainX(currentFrame))
          .attr("y1", -that._mainStep - that._mainMargin.bottom)
          .attr("y2", that._mainHeight);

        if (displayXAxis) {
          focusFrameText.attr("opacity", "1.0");
          focusFrameText.attr("x", d3.pointer(event)[0]);
          focusFrameText.text(currentFrame);
          var textBBox = focusFrameText.node().getBBox();

          focusFrameTextBackground.attr("opacity", "1.0")
          focusFrameTextBackground.attr("x", textBBox.x - textBBox.width / 4);
          focusFrameTextBackground.attr("y", textBBox.y);
          focusFrameTextBackground.attr("width", textBBox.width + textBBox.width / 2);
          focusFrameTextBackground.attr("height", textBBox.height);
          focusFrameTextBackground.attr("fill", "#151b28");
        }

        let idx;

        //focusLineValues.attr("x", d3.pointer(event)[0]);
        focusLineValues.attr("opactiy", "1.0");
        focusLineValues.text(function(d) {
          for (idx = 0; idx < d.graphData.length; idx++) {
            if (d.graphData[idx].frame > currentFrame) {
              if (idx > 0) {
                return d3.format(".2f")(d.graphData[idx - 1].actualValue);
              }
            }
          }
          return "";
        });

        //focusStateValues.attr("x", d3.pointer(event)[0]);
        focusStateValues.attr("opactiy", "1.0");
        focusStateValues.text(function(d) {
          for (idx = 0; idx < d.graphData.length; idx++) {
            if (d.graphData[idx].frame > currentFrame) {
              if (idx > 0) {
                return String(d.graphData[idx - 1].actualValue);
              }
            }
          }
          return "";
        });
    });
  }

  /**
   * Unhighlights all the lines in the main timeline. This is the default.
   */
  _unhighlightMainLines() {
    this._mainLineG.selectAll("use")
      .join("use")
      .attr("opacity", "0.7")
      .attr("stroke", "#797991")
      .attr("stroke-width", 1.0)
      .style("stroke-dasharray", "1, 2");

    this._mainLineText.attr("opacity", "0");
  }

  /**
   * Callback for "end" with d3.brushX
   * @param {array} selection Mouse pointer event
   */
  _mainBrushEnded ({selection}) {
    if (!selection) {
      this._mainBrushG.call(this._mainBrush.move, [-1, -1]);
    }
  }

  /**
   *
   * @param {bool} display True if the main timeline should be displayed. False otherwise.
   */
  showMain(display) {
    if (display) {
      this._mainTimelineDiv.style.display = "block";
    }
    else {
      this._mainTimelineDiv.style.display = "none";
    }
  }

  /**
   *
   * @param {bool} display True if the focus timeline should be displayed. False otherwise.
   */
  showFocus(display) {
    if (display) {
      this._focusTimelineDiv.style.display = "block";
    }
    else {
      this._focusTimelineDiv.style.display = "none";
    }
    this._updateSvgData();
  }

  /**
   *
   */
  updateData(stateData, maxFrame) {

    this._stateData = stateData;
    this._maxFrame = maxFrame;

    this.showMain(true);
    this.showFocus(true);
    this._updateSvgData();
  }
}

customElements.define("fancy-timeline", FancyTimeline);
