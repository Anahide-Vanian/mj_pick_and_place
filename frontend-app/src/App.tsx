import * as Blockly from 'blockly';
// Ensure default blocks and the JavaScript generator are registered
import 'blockly/blocks';
import 'blockly/javascript';

import './App.css'
import { useEffect, useRef, useState } from 'react';

function App() {

  const blocklyDiv = useRef(null as any);
  const workspace = useRef(null as any);
  const [code, setCode] = useState('');


  useEffect(() => {
    if (blocklyDiv.current && !workspace.current) {
      // Define toolbox with available blocks
      const toolbox = {
        kind: 'flyoutToolbox',
        contents: [
          {
            kind: 'block',
            type: 'controls_if'
          },
          {
            kind: 'block',
            type: 'logic_compare'
          },
          {
            kind: 'block',
            type: 'math_number'
          },
          {
            kind: 'block',
            type: 'math_arithmetic'
          },
          {
            kind: 'block',
            type: 'text'
          },
          {
            kind: 'block',
            type: 'text_print'
          },
          {
            kind: 'block',
            type: 'variables_get'
          }
        ]
      };

      console.log('Initializing Blockly workspace');
      console.log(blocklyDiv.current);

      // Initialize workspace
      workspace.current = Blockly.inject(blocklyDiv.current, {
        toolbox: toolbox,
        scrollbars: true,
        trashcan: true,
        zoom: {
          controls: true,
          wheel: true,
          startScale: 1.0,
          maxScale: 3,
          minScale: 0.3,
          scaleSpeed: 1.2
        },
        grid: {
          spacing: 20,
          length: 3,
          colour: '#ccc',
          snap: true
        }
      });

      // Listen for changes
      workspace.current.addChangeListener(() => {
        const generatedCode = Blockly.JavaScript.workspaceToCode(workspace.current);
        setCode(generatedCode);
        console.log(generatedCode)
      });
    }

    // Cleanup
    return () => {
      if (workspace.current) {
        workspace.current.dispose();
        workspace.current = null;
      }
    };
  }, []);

 const handleClear = () => {
    if (workspace.current) {
      workspace.current.clear();
    }
  };
  return (
    <>
     <div className="flex flex-col h-screen bg-gray-100">
      <div className="bg-blue-600 text-white p-4 shadow-lg">
        <h1 className="text-2xl font-bold">Blockly + React Demo</h1>
        <p className="text-sm text-blue-100 mt-1">Drag blocks from the left to build your program</p>
      </div>
      
      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 bg-white border-r border-gray-300">
          {/* Give the Blockly container an explicit size so it is visible even without Tailwind utilities */}
          <div
            ref={blocklyDiv}
            style={{ height: 'calc(100vh - 64px)', width: '100%' }}
          />
        </div>
        
        <div className="w-96 bg-gray-50 p-4 overflow-auto">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-700">Generated Code</h2>
            <button
              onClick={handleClear}
              className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 text-sm"
            >
              Clear Workspace
            </button>
          </div>
          
          <div className="bg-gray-800 text-green-400 p-4 rounded font-mono text-sm overflow-x-auto">
            <pre>{code || '// Your code will appear here'}</pre>
          </div>

        </div>
      </div>
    </div>
    </>
  )
}

export default App
