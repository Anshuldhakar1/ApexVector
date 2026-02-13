This will be a python module "apx-clsct".

## GIT commit

 - Always commit after meaningful changes.
 - Use conventional commits with format ```<type>: <description>```

## Readme 

 - Update with each Git commit. 
 - Should contain the new commands for each mode (pipeline).

## Testing
 - Always run with colors = 24
 - No linter or typechecker is present.
 - Create a tests/ folder
 - Test pipelines using actual images in "../test_images/", test using any 2. Create different folders for each different image you test.
 - USE the svg_to_png convertor to convert the outptus in the  output folder, and improve based on the mistakes in it.

## Building

 - Use the plan present in core.md to implement new code.
 - For this package "apex-clsct" treat "CLSCT" as root directory of the project.
 - Architecture.md is how the user is expected to use this

## Iterative Building

  - Run the updated pipeline whenever to create outputs that the user can see as images and svgs both. 
  - Analyze your own output and create a plan to fix it. Ask the user if it is correct.