package org.janelia.saalfeldlab.conversion;

import java.io.IOException;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5FSReader;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.CacheLoader;
import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.cache.ref.SoftRefLoaderCache;
import net.imglib2.cache.util.LoaderCacheAsCacheAdapter;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.type.label.Label;
import net.imglib2.type.label.LabelMultisetType;
import net.imglib2.type.label.N5CacheLoader;
import net.imglib2.type.label.VolatileLabelMultisetArray;
import net.imglib2.util.Intervals;
import net.imglib2.view.IntervalView;
import net.imglib2.view.Views;
import picocli.CommandLine;
import picocli.CommandLine.Option;

public class LMTLongConverterLegacy
{
	public static void convertLMTToLong( String inputBasePath, String inputDatasetName, String outputBasePath, String outputDatasetName ) throws IOException
	{
		final N5Reader n5reader = new N5FSReader( inputBasePath );
		final N5Writer n5writer = new N5FSWriter( outputBasePath );

		final DatasetAttributes attr = n5reader.getDatasetAttributes( inputDatasetName );

		final long[] dimensions = attr.getDimensions();
		final int[] blocksize = attr.getBlockSize();

		n5writer.createDataset( outputDatasetName, dimensions, blocksize, DataType.UINT64, attr.getCompression() );
		final DatasetAttributes writerAttributes = n5writer.getDatasetAttributes( outputDatasetName );

		final CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > inputImg =
				getSource( new N5CacheLoader( n5reader, inputDatasetName ), dimensions, blocksize );

		final int nDim = dimensions.length;
		final long[] curLocation = new long[nDim];
		
		for(int d = 0; d < nDim; ) {
			
			final long[] actualCellDimensions = IntStream.range( 0, nDim )
					.mapToLong( i -> Math.min( blocksize[ i ], dimensions[ i ] - curLocation[ i ] ) )
					.toArray();

			IntervalView< LabelMultisetType > block = Views.offsetInterval( inputImg, curLocation, actualCellDimensions );
			
			LongArrayDataBlock dataBlock = new LongArrayDataBlock(
					LongStream.of( actualCellDimensions ).mapToInt( i -> ( int ) i ).toArray(),
					IntStream.range( 0, nDim ).mapToLong( i -> curLocation[ i ] / blocksize[ i ] ).toArray(),
					convertBlockToLabels( block, ( int ) Intervals.numElements( block ) ) );
			
			n5writer.writeBlock( outputDatasetName, writerAttributes, dataBlock );

			for(d = 0; d < nDim; d++) {
				curLocation[d] += blocksize[d];
				if(curLocation[d] < dimensions[d])
					break;
				else
					curLocation[d] = 0;
			}
		}
	}

	public static long[] convertBlockToLabels( final RandomAccessibleInterval< LabelMultisetType > source, int numElements )
	{
		long[] labels = new long[ numElements ];
		int idx = 0;
		for ( LabelMultisetType lmt : Views.flatIterable( source ) )
		{
			long tmpLabel = lmt.entrySet().iterator().next().getElement().id();
			labels[ idx++ ] = tmpLabel; // lmt.entrySet().iterator().next().getElement().id();
		}
		return labels;
	}

	public static class CommandLineParameters
	{
		@Option( names = { "-i", "--input-n5" }, description = "Input N5 container", required = true )
		private String inputN5Root;

		@Option( names = { "-id", "--input-dataset" }, description = "Input dataset path relative to input N5 root", required = true )
		private String inputDataset;

		@Option( names = { "-o", "--output-n5" }, description = "Output N5 container", required = true )
		private String outputN5Root;

		@Option( names = { "-od", "--output-dataset" }, description = "Output dataset path relative to output N5 root" )
		private String outputDataset;

		@Option( names = { "-h", "--help" }, usageHelp = true, description = "Display a help message" )
		private boolean helpRequested;
	}

	public static void main( String[] args ) throws IOException
	{
		final CommandLineParameters clp = new CommandLineParameters();
		final CommandLine cl = new CommandLine( clp );
		cl.parse( args );

		if ( cl.isUsageHelpRequested() )
		{
			cl.usage( System.out );
			return;
		}

		convertLMTToLong( clp.inputN5Root, clp.inputDataset, clp.outputN5Root, ( clp.outputDataset == null ? clp.inputDataset : clp.outputDataset ) );
	}


	public static CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > getSource(
			final CacheLoader< Long, Cell< VolatileLabelMultisetArray > > cacheLoader,
			final long[] dimensions,
			final int[] blockSize )
	{
		final SoftRefLoaderCache< Long, Cell< VolatileLabelMultisetArray > > cache = new SoftRefLoaderCache<>();
		final LoaderCacheAsCacheAdapter< Long, Cell< VolatileLabelMultisetArray > > wrappedCache = new LoaderCacheAsCacheAdapter<>( cache, cacheLoader );

		final CachedCellImg< LabelMultisetType, VolatileLabelMultisetArray > source = new CachedCellImg<>(
				new CellGrid( dimensions, blockSize ),
				new LabelMultisetType().getEntitiesPerPixel(),
				wrappedCache,
				new VolatileLabelMultisetArray( 0, true, new long[] { Label.INVALID } ) );
		source.setLinkedType( new LabelMultisetType( source ) );
		source.getCellGrid();
		return source;
	}

}
